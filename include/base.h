#pragma once
#include <vector>
#include "matrix.h"

class Layer_Base{
    
    public:

    Matrix<float> inputs;
    Matrix<float> output;
    Matrix<float> weights;
    Matrix<float> bias;
    Matrix<float> weight_momentums;
    Matrix<float> bias_momentums;
    Matrix<float> weight_cache;
    Matrix<float> bias_cache;
    Matrix<float> dl_dx;
    Matrix<float> dl_dw;
    Matrix<float> dl_db;
    bool require_grad;

    Layer_Base();
    virtual ~Layer_Base();
    virtual Matrix<float> forward(const Matrix<float>&) = 0;
    virtual void backward(const Matrix<float>&) = 0;

};

class Optimizer_Base{
    
    protected:

    float learning_rate;
    float curr_learning_rate;
    float decay;
    int iterations;

    void pre_update_params();
    virtual void update_params(std::vector<Layer_Base*>& trainable_layers) const = 0;
    void post_update_params();

    public:

    Optimizer_Base(float learning_rate, float decay);
    virtual ~Optimizer_Base();

    void zero_grad(std::vector<Layer_Base*>& trainable_layers);
    void step(std::vector<Layer_Base*>& trainable_layers);

};

class Loss_Base{

    protected:

    Matrix<float> dl_dx;
    float loss;

    public:

    virtual ~Loss_Base();
    virtual float forward(const Matrix<float>& y_pred, const Matrix<float>& y_true) const = 0;
    virtual void backward(const Matrix<float>& y_pred, const Matrix<float>& y_true) = 0;

    void step(std::vector<Layer_Base*>& trainable_layers, Matrix<float>& y_pred, Matrix<float>& y_true);
    float get_loss();
    
};