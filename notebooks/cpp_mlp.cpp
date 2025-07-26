// cpp_mlp.cpp

#include <torch/extension.h>   // 包含 <torch/torch.h> + 自动注册 at::Tensor ↔ Python torch.Tensor
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <algorithm>
#include <sstream>

namespace py = pybind11;

// 简单的 split 工具
static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

struct MLP : torch::nn::Module {
    torch::nn::Linear      fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr};
    torch::nn::Dropout     dropout{nullptr};

    MLP(int64_t in, int64_t h1, int64_t h2, int64_t out) {
        std::cout << "[MLP] 模型构建开始: "
                  << "in=" << in << " h1=" << h1
                  << " h2=" << h2 << " out=" << out << std::endl;

        if (in <= 0 || h1 <= 0 || h2 <= 0 || out <= 0)
            throw std::invalid_argument("所有参数要为正值");

        fc1     = register_module("fc1",     torch::nn::Linear(in,  h1));
        bn1     = register_module("bn1",     torch::nn::BatchNorm1d(h1));
        fc2     = register_module("fc2",     torch::nn::Linear(h1, h2));
        bn2     = register_module("bn2",     torch::nn::BatchNorm1d(h2));
        fc3     = register_module("fc3",     torch::nn::Linear(h2, out));
        dropout = register_module("dropout", torch::nn::Dropout(0.5));

        std::cout << "[MLP] 模型构建完毕" << std::endl;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1(fc1(x)));
        x = dropout(x);
        x = torch::relu(bn2(fc2(x)));
        x = dropout(x);
        return fc3(x);
    }
};

PYBIND11_MODULE(cpp_mlp, m) {
    m.doc() = "cpp_mlp PyTorch C++ extension";

    py::class_<MLP, std::shared_ptr<MLP>>(m, "MLP")
        // 构造函数
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(),
             py::arg("input_dim"),
             py::arg("hidden_dim1"),
             py::arg("hidden_dim2"),
             py::arg("output_dim"))

        // forward: Python 调用 model.forward(x)
        .def("forward", &MLP::forward, py::arg("x"))

        // __call__: 使 model(x) 等价于 model.forward(x)
        .def("__call__", &MLP::forward, py::arg("x"))

        // 切换到训练/评估模式，并支持链式调用
        .def("train",
             [](std::shared_ptr<MLP> self, bool on = true) {
                 self->train(on);
                 return self;
             },
             py::arg("on") = true,
             py::return_value_policy::reference)

        .def("eval",
             [](std::shared_ptr<MLP> self) {
                 self->eval();
                 return self;
             },
             py::return_value_policy::reference)

        // to(device): 支持字符串或 torch.device 对象
        .def("to",
             [](std::shared_ptr<MLP> self, py::object dev_obj) {
                 torch::Device device(torch::kCPU, 0);

                 if (py::isinstance<py::str>(dev_obj)) {
                     std::string ds = dev_obj.cast<std::string>();
                     std::transform(ds.begin(), ds.end(), ds.begin(), ::tolower);
                     auto parts = split(ds, ':');
                     if (parts[0] == "cpu") {
                         device = torch::Device(torch::kCPU);
                     } else if (parts[0] == "cuda" || parts[0] == "gpu") {
                         int idx = (parts.size() > 1 ? std::stoi(parts[1]) : 0);
                         device = torch::Device(torch::kCUDA, idx);
                     } else {
                         throw std::runtime_error("错误的设备参数: " + ds);
                     }
                 }
                 else if (py::hasattr(dev_obj, "type") && py::hasattr(dev_obj, "index")) {
                     std::string tp = dev_obj.attr("type").cast<std::string>();
                     std::transform(tp.begin(), tp.end(), tp.begin(), ::tolower);
                     int idx = dev_obj.attr("index").is_none()
                               ? 0 : dev_obj.attr("index").cast<int>();
                     if (tp == "cpu") {
                         device = torch::Device(torch::kCPU);
                     } else if (tp == "cuda") {
                         device = torch::Device(torch::kCUDA, idx);
                     } else {
                         throw std::runtime_error("不支持的设备类型: " + tp);
                     }
                 }
                 else {
                     throw std::runtime_error(
                         "to() 方法需要字符串输入或者设备类型输入 "
                         + std::string(py::str(dev_obj)));
                 }

                 self->to(device);
                 return self;
             },
             py::arg("device"),
             py::return_value_policy::reference)

        // parameters(): 屏蔽 bool 参数，Python 直接调用无参
        .def("parameters",
             [](std::shared_ptr<MLP> self) {
                 return self->parameters();
             },
             py::return_value_policy::copy);

    // 可选：一键训练与评估函数
    m.def("train_and_evaluate",
          [](torch::Tensor inputs,
             torch::Tensor labels,
             std::shared_ptr<MLP> model,
             int64_t epochs = 10,
             double lr = 0.001) {
              model->train();
              auto criterion = torch::nn::CrossEntropyLoss();
              auto optimizer = torch::optim::Adam(model->parameters(), lr);
              for (int64_t e = 0; e < epochs; ++e) {
                  optimizer.zero_grad();
                  auto out = model->forward(inputs);
                  auto loss = criterion(out, labels);
                  loss.backward();
                  optimizer.step();
                  std::cout << "[Epoch " << (e + 1)
                            << "] 损失: " << loss.item<float>()
                            << std::endl;
              }
          },
          py::arg("inputs"),
          py::arg("labels"),
          py::arg("model"),
          py::arg("epochs") = 10,
          py::arg("lr") = 0.001);
}