# Source and third-party notices

This package extracts image-processing code from CyperstereoSDK and the local
`camera_tuner` application. This notice records the observed sources and
file-level notices. It does not replace those notices or grant a new uniform
license to all bundled components.

## CyperstereoSDK source snapshot

- Source repository: <https://github.com/Cyperstereo/CyperstereoSDK>
- Local source checkout: `/home/taoqiu/CyperstereoSDK`
- Git HEAD when inspected: `764ab5497789172fd5686d771dde83e8533d22cf`
- Extraction audit date: 2026-10-16 (Asia/Shanghai)
- The repository's root `LICENSE` contains the GNU General Public License,
  version 3, dated 29 June 2007. That file is preserved as this package's
  [LICENSE](LICENSE).
- The `camera_tuner` directory was local, untracked work at extraction time.
  Its source must not be represented as part of the Git commit above.
  [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records the copied source paths
  and snapshot hashes. Package-specific adapters and edits must be distinguished
  from their source snapshots when comparing those hashes.

Original source comments, authorship notices, version markers, and individual
license headers are retained in the copied files.

## UVC declarations and ISP integration

The material under `vendor/uvc/` originated in `src/usb/uvc/` of the source
checkout. In particular, `uvc.h` and `types.h` carry these original notices:

> Copyright 2018 Slightech Co., Ltd. All rights reserved.

Those files explicitly identify the Apache License, Version 2.0. Its full text
is included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt), copied from the
host's `/usr/share/common-licenses/Apache-2.0`. The original source links to
<https://www.apache.org/licenses/LICENSE-2.0>.

Other copied SDK headers do not all contain that Apache notice. The license
statement of `uvc.h` or `types.h` must not be generalized to every SDK or ISP
file in this package.

The copied `bayer_format.h` and `smartsens_metadata.h` preserve SmartSens
hardware/firmware identifiers and CFA mapping information. They are functional
version data, not a claim that the package was tested with every corresponding
camera or firmware.

## ISP modules

The material under `vendor/isp/` originated in `src/ISP/srcs/`. Many original
headers identify `joker.mao` and contain `Copyright (c) of ADAS_EYES 2023`, with
individual file versions such as `0.1`. The integration source also describes
its relationship to HDR-ISP-main. These source notices are retained verbatim.

No separate license file was found inside the source checkout's `src/ISP/`
module directory. This notice therefore does not assign a new MIT, BSD, or
Apache license to these modules; it records the source repository license and
the notices actually present in the extracted files.

`src/isp_frame_storage.cpp` is the local `camera_tuner` storage implementation
used in place of the original ISP `frame.cpp`. It preserves the SDK `Frame`
layout while providing typed array deletion, initialized buffers, and cleanup
on allocation failure. The original and replacement implementations must not
both be linked into the same target.

## Easylogging++

- Bundled files: `vendor/easylog/easylogging++.h` and `easylogging++.cc`.
- Version stated by both bundled file headers: **9.97.1**.
- Copyright stated by those headers and the corresponding upstream license:
  `Copyright (c) 2012-present @abumq (Majid Q.)`.
- License stated by the headers: **MIT**.
- Full license: [LICENSES/Easyloggingpp-MIT.txt](LICENSES/Easyloggingpp-MIT.txt).
- Verified upstream license for this version:
  <https://raw.githubusercontent.com/abumq/easyloggingpp/v9.97.1/LICENSE>
  (retrieved during the 2026-10-16 extraction audit).

The copyright wording above is preserved exactly; `present` has not been
replaced with a guessed year. The version-specific upstream license was used
rather than the changing `master` branch URL mentioned in the bundled headers.

## External build dependencies and excluded components

The processing code uses C++17, the C++ standard library, threading support, and
OpenCV core/image-processing APIs. File-based examples or configuration helpers
may additionally use OpenCV image codecs and persistence APIs. OpenCV and its
binary dependencies are supplied by the consumer's build environment; their
own distribution notices and licenses remain applicable.

The original GUI's Qt sources and camera capture implementation are separate
from the image-processing code. UVC declarations are needed by the inherited
SDK header, but the platform UVC backends and `libusb` are not needed merely to
process already acquired Bayer frames. Qt-based configuration I/O must not be
introduced into the portable core; the local application implementation uses
`QFile` and `QSaveFile` and is not itself a Qt-free dependency.

EasyBMP, nlohmann/json, original standalone ISP programs, sample images, and
camera capture backends are not part of this extracted dependency set. Their
source directories are not required by the pure ISP library.
