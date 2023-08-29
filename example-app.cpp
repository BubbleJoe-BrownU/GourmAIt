#include <torch/extension.h>
#include <torch/script.h>

#include <iostream>
#include <memory>

torch::Tensor trilinear_interpolation(
    torch::Tensor feats,
    torch::Tensor point
) {
    return feats;
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the module\n";
        return -1;
    }

    std::cout << "ok\n";
}