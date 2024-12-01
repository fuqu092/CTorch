#pragma once
#include "base.h"

class Activation_ReLU : public Layer_Base{

    public:

    Activation_ReLU();
    ~Activation_ReLU() override;

    Activation_ReLU(const Activation_ReLU&) = delete;
    Activation_ReLU& operator=(const Activation_ReLU&) = delete;
    Activation_ReLU(Activation_ReLU&&) = delete;
    Activation_ReLU& operator=(Activation_ReLU&&) = delete;

    Matrix<float> forward(const Matrix<float>& inputs) override;
    void backward(const Matrix<float>& dl_dz) override;

};