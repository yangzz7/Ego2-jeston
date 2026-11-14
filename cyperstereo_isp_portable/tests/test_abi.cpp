#include <cyperstereo_isp/c_api.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    void* handle = nullptr;
    try {
        if (cisp_api_version() != 1) throw std::runtime_error("ABI version");
        if (cisp_create("/missing/cisp-config.yaml")) throw std::runtime_error("Missing config accepted");
        if (!std::strlen(cisp_last_error())) throw std::runtime_error("Missing error text");
        handle = cisp_create(nullptr);
        if (!handle) throw std::runtime_error(cisp_last_error());
        constexpr size_t bytes = 1280 * 1024 * 3;
        std::vector<uint8_t> a(bytes, 123), b(bytes, 45), raw(1280 * 1024 * 2);
        if (cisp_process_transport(handle, raw.data(), raw.size(), 2560,
                                  a.data(), bytes, 3840, b.data(), bytes, 3840, nullptr) != -1)
            throw std::runtime_error("Invalid firmware accepted");
        if (a.front() != 123 || a.back() != 123 || b.front() != 45) throw std::runtime_error("Failure changed outputs");
        if (cisp_process_transport(handle, raw.data(), raw.size(), 2560,
                                  a.data(), 1, 3840, b.data(), bytes, 3840, nullptr) != -1)
            throw std::runtime_error("Short output accepted");
        if (cisp_reset(nullptr) != -1 || cisp_load_config(handle, nullptr) != -1)
            throw std::runtime_error("Null argument accepted");
        if (cisp_reset(handle) != 0) throw std::runtime_error(cisp_last_error());
        cisp_destroy(handle); handle = nullptr; cisp_destroy(nullptr);
        std::cout << "C ABI checks passed\n";
    } catch (const std::exception& e) {
        cisp_destroy(handle); std::cerr << e.what() << '\n'; return 1;
    }
}
