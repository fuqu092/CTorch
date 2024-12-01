#pragma once
#include "base.h"

class Layer_Dense : public Layer_Base{

    private:

    float weight_regularizer_l1;
    float weight_regularizer_l2;
    float bias_regularizer_l1;
    float bias_regularizer_l2;

    public:

    ~Layer_Dense() override;

    Layer_Dense(const Layer_Dense&) = delete;
    Layer_Dense& operator=(const Layer_Dense&) = delete;
    Layer_Dense(Layer_Dense&&) = delete;
    Layer_Dense& operator=(Layer_Dense&&) = delete;

    Layer_Dense(int n_inputs, int n_neurons, float weight_l1, float weight_l2, float bias_l1, float bias_l2);

    Matrix<float> forward(const Matrix<float>& inputs) override;

    void backward(const Matrix<float>& dl_dz) override;

};