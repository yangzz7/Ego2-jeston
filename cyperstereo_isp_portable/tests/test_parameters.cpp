#include "camera_tuner/model.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __linux__
#include <csignal>
#include <sys/resource.h>
#endif

using namespace camera_tuner;
namespace fs = std::filesystem;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void rejects(const std::function<void()>& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected,"Invalid parameters or failed I/O were accepted");
}

struct TemporaryDirectory {
    fs::path path;
    TemporaryDirectory() {
        std::random_device random;
        const auto seed = std::to_string(random()) + "-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        for (int i = 0; i < 64; ++i) {
            auto candidate = fs::temp_directory_path() / ("cyper-parameters-test-" + seed + "-" + std::to_string(i));
            if (fs::create_directory(candidate)) { path = std::move(candidate); return; }
        }
        throw std::runtime_error("Cannot create a unique temporary test directory");
    }
    ~TemporaryDirectory() { std::error_code ignored; fs::remove_all(path,ignored); }
};

std::string read(const fs::path& path) {
    std::ifstream stream(path,std::ios::binary);
    require(bool(stream),"Cannot read test fixture");
    std::string bytes{std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
    require(!stream.bad(),"Fixture read failed");
    return bytes;
}

void write(const fs::path& path, const std::string& bytes) {
    std::ofstream stream(path,std::ios::binary | std::ios::trunc);
    require(bool(stream),"Cannot write test fixture");
    stream.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));
    stream.flush();
    require(bool(stream),"Fixture write was incomplete");
}

std::vector<std::string> entries(const fs::path& directory) {
    std::vector<std::string> result;
    for (const auto& entry : fs::directory_iterator(directory)) result.push_back(entry.path().filename().string());
    std::sort(result.begin(),result.end());
    return result;
}

// Match complete keys, so white_balance never replaces auto_white_balance.
std::string editLine(std::string text, const std::string& field, const std::string& value, bool remove = false) {
    for (std::size_t begin = 0; begin < text.size();) {
        const auto newline = text.find('\n',begin);
        const auto end = newline == std::string::npos ? text.size() : newline;
        const auto key = text.find_first_not_of(" \t",begin);
        if (key < end && text.compare(key,field.size()+1,field+":") == 0) {
            if (remove) text.erase(begin,end-begin+(newline != std::string::npos));
            else text.replace(key,end-key,field+": "+value);
            return text;
        }
        begin = end+1;
    }
    throw std::runtime_error("Cannot find fixture key " + field);
}

void verifyRangeValidation(const Parameters& valid, const fs::path& existing, const std::string& original) {
    const std::pair<double EyeParameters::*,std::pair<double,double>> fields[] = {
        {&EyeParameters::blackLevel,{0,64}}, {&EyeParameters::exposureEv,{-3,3}},
        {&EyeParameters::gamma,{1,4}}, {&EyeParameters::blackPoint,{0,32}},
        {&EyeParameters::contrast,{0,1}}, {&EyeParameters::saturation,{0,2}},
        {&EyeParameters::bayerDenoise,{0,1}}, {&EyeParameters::lumaDenoise,{0,1}},
        {&EyeParameters::chromaDenoise,{0,1}}, {&EyeParameters::falseColor,{0,1}},
        {&EyeParameters::sharpening,{0,1}},
    };
    for (const auto& field : fields) {
        for (double value : {field.second.first-.01,field.second.second+.01,
                             std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}) {
            auto invalid = valid;
            invalid.eyes[1].*field.first = value;
            rejects([&] { saveParameters(existing.string(),invalid); });
            require(read(existing) == original,"Invalid save truncated the existing configuration");
        }
    }
    auto invalid = valid;
    invalid.schemaVersion = 2;
    rejects([&] { validateParameters(invalid); });
    invalid = valid; invalid.mode = static_cast<PipelineMode>(99);
    rejects([&] { validateParameters(invalid); });
    invalid = valid; invalid.eyes[0].tone = static_cast<ToneMode>(99);
    rejects([&] { validateParameters(invalid); });
    invalid = valid; invalid.eyes[0].whiteBalance[0] = .2;
    rejects([&] { validateParameters(invalid); });
    invalid = valid; invalid.eyes[0].whiteBalance[2] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { validateParameters(invalid); });
    invalid = valid; invalid.eyes[1].ccm[4] = 4.01;
    rejects([&] { validateParameters(invalid); });
    invalid = valid; invalid.serial = "../s200032";
    rejects([&] { validateParameters(invalid); });
}

void verifyAtomicFailure(const fs::path& root, const Parameters& parameters,
                         const fs::path& existing, const std::string& original) {
    const auto blocker = root / "destination-directory";
    fs::create_directory(blocker);
    write(blocker/"keep.txt","user data");
    const auto before = entries(root);
    rejects([&] { saveParameters(blocker.string(),parameters); });
    require(entries(root) == before,"Failed rename left a temporary file");
    require(read(blocker/"keep.txt") == "user data","Failed rename changed existing user data");
    rejects([&] { saveParameters((root/"missing"/"config.yaml").string(),parameters); });
    require(entries(root) == before && read(existing) == original,"Failed save changed existing files");

#ifdef __linux__
    // Force an actual short write/flush failure, not just a validation error.
    // This modifies only the test process, and restores its soft resource limit.
    struct FileLimit {
        rlimit old{};
        using Handler = void(*)(int);
        Handler previous = SIG_DFL;
        FileLimit() {
            require(getrlimit(RLIMIT_FSIZE,&old) == 0,"Cannot read test file-size limit");
            previous = std::signal(SIGXFSZ,SIG_IGN);
            require(previous != SIG_ERR,"Cannot ignore test SIGXFSZ");
            rlimit limited = old;
            limited.rlim_cur = 128;
            if (setrlimit(RLIMIT_FSIZE,&limited) != 0) {
                std::signal(SIGXFSZ,previous);
                throw std::runtime_error("Cannot set test file-size limit");
            }
        }
        ~FileLimit() { setrlimit(RLIMIT_FSIZE,&old); std::signal(SIGXFSZ,previous); }
    };
    {
        FileLimit limit;
        rejects([&] { saveParameters(existing.string(),parameters); });
    }
    require(read(existing) == original,"A short write replaced the existing configuration");
    require(entries(root) == before,"A short write left a temporary parameters file");
#endif
}

void runTests(const fs::path& defaultConfig) {
    const auto defaults = loadParameters(defaultConfig.string());
    require(defaults.serial == "s200032" && defaults.mode == PipelineMode::Tunable,
            "Shipped tuner-compatible defaults did not load");
    require(defaults.eyes[0].tone == ToneMode::SRgb && defaults.eyes[0].blackLevel == 16 &&
            defaults.eyes[1].whiteBalance == std::array<double,3>{{1,1,1}},"Default eye parameters changed");
    TemporaryDirectory directory;
    const auto path = directory.path/"parameters.yaml";
    Parameters parameters = defaults;
    parameters.linked = false;
    parameters.mode = PipelineMode::Minimal;
    parameters.eyes[0].autoWhiteBalance = false;
    parameters.eyes[0].whiteBalance = {{1.5,1.25,.75}};
    parameters.eyes[0].ccmEnabled = true;
    parameters.eyes[0].ccm = {{1.7,-.6,-.1,-.2,1.4,-.2,0,-.5,1.5}};
    parameters.eyes[0].exposureEv = -1.25;
    parameters.eyes[1].tone = ToneMode::Power;
    parameters.eyes[1].gamma = 3.1;
    parameters.eyes[1].sharpening = .3;
    parameters.revision = 450;
    saveParameters(path.string(),parameters);
    const auto original = read(path);
    const auto loaded = loadParameters(path.string());
    require(loaded.serial == parameters.serial && loaded.mode == PipelineMode::Minimal && !loaded.linked &&
            loaded.revision == 1,"Configuration header or revision round trip failed");
    require(loaded.eyes[0].whiteBalance == parameters.eyes[0].whiteBalance &&
            loaded.eyes[0].ccm == parameters.eyes[0].ccm && loaded.eyes[0].exposureEv == -1.25,
            "First eye gains/matrix/exposure did not round trip");
    require(loaded.eyes[1].tone == ToneMode::Power && loaded.eyes[1].gamma == 3.1 &&
            loaded.eyes[1].sharpening == .3,"Second eye did not round trip");
    const auto roundtrip = directory.path/"roundtrip.yaml";
    saveParameters(roundtrip.string(),loaded);
    require(read(roundtrip) == original,"Round trip changed the serialized configuration");
    // Successfully replacing an existing file must also work, without delete.
    saveParameters(path.string(),defaults);
    require(loadParameters(path.string()).mode == PipelineMode::Tunable,"Atomic replacement failed");
    saveParameters(path.string(),parameters);
    require(read(path) == original,"Replacing the saved configuration lost values");

    const auto badPath = directory.path/"invalid.yaml";
    const std::pair<std::string,std::string> badValues[] = {
        {"schema_version","2"},{"mode","unknown"},{"linked","2"},{"linked","0.0"},
        {"linked","true"},{"serial","200032"},{"tone","mystery"},
        {"auto_white_balance","2"},{"gamma",".Nan"},{"gamma",".Inf"},{"gamma","5.0"},
    };
    for (const auto& change : badValues) {
        write(badPath,editLine(original,change.first,change.second));
        rejects([&] { loadParameters(badPath.string()); });
    }
    write(badPath,editLine(original,"gamma","",true));
    rejects([&] { loadParameters(badPath.string()); });
    write(badPath,editLine(original,"schema_version","1\nunknown_field: 9"));
    rejects([&] { loadParameters(badPath.string()); });
    write(badPath,editLine(original,"linked","0\nlinked: 0"));
    rejects([&] { loadParameters(badPath.string()); });
    write(badPath,"%YAML:1.0\n---\nschema_version: 1\nserial: s200032\nmode: tunable\nlinked: 1\neyes: []\n");
    rejects([&] { loadParameters(badPath.string()); });
    write(badPath,std::string(1024*1024+1,'x'));
    rejects([&] { loadParameters(badPath.string()); });
    rejects([&] { loadParameters((directory.path/"not-found.yaml").string()); });
    verifyRangeValidation(parameters,path,original);
    verifyAtomicFailure(directory.path,parameters,path,original);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto defaults = argc > 1 ? fs::path(argv[1]) :
            fs::path(__FILE__).parent_path().parent_path()/"configs"/"s200032.yaml";
        runTests(defaults);
        std::cout << "Portable parameter schema, round-trip and atomic-save checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
