"""Lossless SmartSens S2 metadata capture with the bundled SDK's conversions.

Layouts and heuristics come from ``smartsens_metadata.h`` and
``cyperstereo_api.h`` in third_party/orb_slam3/Thirdparty/usb/uvc. Device
timestamps are counters, not UTC or host monotonic time. IMU acceptance is
the SDK's continuity heuristic; retained raw slots allow later reprocessing.

Firmware 07 follows CyperstereoSDK/src/usb/uvc/smartsens_metadata.h
(verified 2026-10-24): firmware-06 IMU/AE timing plus GNSS words 81..105.
GNSS is preserved in the raw row; this adapter does not decode GNSS fixes.
"""

import math

import numpy as np


ACCEL_G_PER_COUNT = 0.00018310546875  # BMI088, +/-6 g, SDK default
GYRO_RAD_S_PER_COUNT = 0.0010652644178602  # BMI088, +/-2000 deg/s
STANDARD_GRAVITY_M_S2 = 9.80665
PERIOD_SECONDS = 43200

# slots, AE marker column, exposure row time in seconds
_LAYOUTS = {
    3: (7, None, 23.868131868e-6),
    4: (7, 68, 23.868131868e-6),
    5: (13, 122, 23.572016461e-6),
    6: (7, 68, 452.0 / 15187500.0),
    7: (7, 68, 452.0 / 15187500.0),
}


def metadata_words(frame, width, height):
    """Copy every last-row little-endian word before image reconstruction."""
    if frame.dtype != np.uint8 or frame.size != width * height * 2:
        raise ValueError("Expected raw packed Cyperstereo YUYV transport bytes; RGB conversion must be disabled")
    if width < 2 or height < 3:
        raise ValueError("Cyperstereo transport is too small for metadata/Bayer reconstruction")
    row = frame.reshape(height, width, 2)[-1]
    return row[:, 0].astype(np.uint16) | (row[:, 1].astype(np.uint16) << 8)


def _signed(word):
    return word - 65536 if word >= 32768 else word


class CyperMetadataDecoder:
    """Decode all slots, maintaining only the SDK's temperature reference."""

    def __init__(self, acceleration_scale=STANDARD_GRAVITY_M_S2):
        self.acceleration_scale = float(acceleration_scale)
        if not math.isfinite(self.acceleration_scale) or self.acceleration_scale <= 0:
            raise ValueError("input.acceleration_scale must be finite and positive (m/s² per g)")
        self.last_imu_temperature = None
        self.firmware = None

    def decode(self, frame, width, height):
        return self.decode_words(metadata_words(frame, width, height))

    def decode_words(self, words):
        words = np.asarray(words)
        if words.ndim != 1 or words.size < 2 or words.dtype.kind not in "iu":
            raise ValueError("Metadata must be a one-dimensional row of unsigned 16-bit words")
        if np.any(words < 0) or np.any(words > 65535):
            raise ValueError("Metadata words must be in [0, 65535]")
        raw = [int(word) for word in words]
        hardware, software = raw[:2]
        if hardware != 2 or software not in _LAYOUTS:
            raise ValueError(f"Unsupported/invalid SmartSens metadata {hardware}/{software}; expected S2 firmware 03..07")
        slot_count, ae_col, line_time = _LAYOUTS[software]
        end_col = ae_col + 13 if ae_col is not None else 5 + 9 * slot_count
        if software == 7:
            end_col = 106  # Include the GNSS extension, retained as raw words.
        if len(raw) < end_col:
            raise ValueError(f"Truncated SmartSens {hardware}/{software} metadata: need {end_col} words")
        if self.firmware != (hardware, software):
            self.last_imu_temperature = None
            self.firmware = (hardware, software)

        period, ticks, seconds = raw[2:5]
        image_timestamp = period * PERIOD_SECONDS + seconds + ticks / 10000.0
        timestamp_fields_in_range = ticks < 10000 and seconds < PERIOD_SECONDS
        slots, accepted = [], []
        previous_timestamp = None
        previous_temperature = self.last_imu_temperature
        stopped = False
        for index in range(slot_count):
            sample_words = raw[5 + 9 * index:5 + 9 * (index + 1)]
            imu_ticks, imu_seconds = sample_words[:2]
            timestamp = period * PERIOD_SECONDS + imu_seconds + imu_ticks / 10000.0
            # The image and its IMU samples may lie on opposite sides of the
            # 12-hour rollover. The image contributes the missing period field.
            if timestamp - image_timestamp > PERIOD_SECONDS / 2:
                timestamp -= PERIOD_SECONDS
            elif image_timestamp - timestamp > PERIOD_SECONDS / 2:
                timestamp += PERIOD_SECONDS
            signed_motion = [_signed(value) for value in sample_words[2:8]]
            temperature_raw = _signed(sample_words[8])
            if temperature_raw > 1023:
                temperature_raw -= 2048
            temperature = temperature_raw * 0.125 + 23
            acceleration_g = [value * ACCEL_G_PER_COUNT for value in signed_motion[:3]]
            sample = {
                "slot_index": index,
                "raw_words": sample_words,
                "timestamp_raw": {"subsecond_ticks": imu_ticks, "seconds": imu_seconds},
                "timestamp_seconds": timestamp,
                "acceleration_raw": signed_motion[:3],
                "angular_velocity_raw": signed_motion[3:],
                "temperature_raw": sample_words[8],
                "acceleration_g": acceleration_g,
                "acceleration_m_s2": [value * self.acceleration_scale for value in acceleration_g],
                "angular_velocity_rad_s": [value * GYRO_RAD_S_PER_COUNT for value in signed_motion[3:]],
                "temperature_c": temperature,
            }
            reason = None
            if stopped:
                reason = "after_rejected_slot"
            elif software == 5 and not any(sample_words):
                reason = "zero_filled_slot"
            elif not timestamp_fields_in_range:
                reason = "image_timestamp_fields_out_of_range"
            elif abs(timestamp - image_timestamp) > 0.5:
                reason = "timestamp_far_from_image"
            elif (previous_timestamp is not None and previous_timestamp > 0
                  and not 0 < timestamp - previous_timestamp <= 0.050):
                reason = "timestamp_discontinuity"
            elif not -45 <= temperature <= 130:
                reason = "temperature_out_of_range"
            elif previous_temperature is not None and abs(temperature - previous_temperature) > 8:
                reason = "temperature_discontinuity"
            sample["accepted"] = reason is None
            sample["rejection_reason"] = reason
            slots.append(sample)
            if reason is None:
                accepted.append(sample)
                previous_timestamp, previous_temperature = timestamp, temperature
            else:
                stopped = True
        if accepted:
            self.last_imu_temperature = previous_temperature

        telemetry = []
        if ae_col is not None:
            # Python's image order is low lane C1=A, high lane C2=B. It is
            # opposite the C++ left/right display order, not physical order.
            for sensor in range(2):
                exposure = raw[ae_col + 1 + sensor]
                temperature = raw[ae_col + 5 + sensor]
                gain = raw[ae_col + 9 + sensor]
                exposure_seconds = exposure * line_time
                telemetry.append({
                    "sensor": f"C{sensor + 1}",
                    "image_index": sensor,
                    "exposure_lines": exposure,
                    "exposure_seconds": exposure_seconds,
                    "temperature_raw": temperature,
                    "temperature_c": ((temperature >> 8) * 8 + (temperature & 7)) / 4.0 - 273.15,
                    "gain_raw": gain,
                    "gain": gain / 64.0,
                    "estimated_midpoint_timestamp_seconds": image_timestamp - exposure_seconds / 2,
                })
        return {
            "firmware": {"hardware": hardware, "software": software},
            "raw_metadata_words": raw,
            "metadata_end_column_exclusive": end_col,
            "timestamp_domain": "device_counter_seconds",
            "image_timestamp_raw": {"period_12h": period, "subsecond_ticks": ticks, "seconds": seconds},
            "image_timestamp_seconds": image_timestamp,
            "image_timestamp_fields_in_range": timestamp_fields_in_range,
            "imu_acceptance_method": "bundled_sdk_continuity_heuristic",
            "acceleration_scale_m_s2_per_g": self.acceleration_scale,
            "imu_slot_count": slot_count,
            "imu_samples": accepted,
            "imu_slots": slots,
            "telemetry_available": ae_col is not None,
            "telemetry_marker_raw": raw[ae_col] if ae_col is not None else None,
            # The SDK documents the layout but does not specify/check an AE
            # marker value. Do not invent a hardware validity assertion.
            "telemetry_marker_validated": False,
            "telemetry_source": "controller_target_state_without_register_readback",
            "telemetry": telemetry,
            "gnss_available": False,
        }
