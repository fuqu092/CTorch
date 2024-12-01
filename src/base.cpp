#include "base.h"

Layer_Base::Layer_Base() = default;
Layer_Base::~Layer_Base() = default;


Optimizer_Base::Optimizer_Base(float learning_rate, float decay) : learning_rate{learning_rate}, curr_learning_rate{learning_rate}, decay{decay}, iterations{0}{}

Optimizer_Base::~Optimizer_Base() = default;

void Optimizer_Base::pre_update_params(){
    this->curr_learning_rate = this->learning_rate / (1.0 + (this->decay * this->iterations));
}

void Optimizer_Base::post_update_params(){
    this->iterations++;
}

void Optimizer_Base::zero_grad(std::vector<Layer_Base*>& trainable_layers){
    for(auto& layer : trainable_layers){
        layer->dl_dx.set_zero();

        if(!layer->require_grad)
            continue;

        layer->dl_dw.set_zero();
        layer->dl_db.set_zero();
    }
    
    return ;
}

void Optimizer_Base::step(std::vector<Layer_Base*>& trainable_layers){
    this->pre_update_params();
    this->update_params(trainable_layers);
    this->post_update_params();
}


Loss_Base::~Loss_Base() = default;

void Loss_Base::step(std::vector<Layer_Base*>& trainable_layers, Matrix<float>& y_pred, Matrix<float>& y_true){
    this->loss = this->forward(y_pred, y_true);
    this->backward(y_pred, y_true);
    Matrix<float> temp = this->dl_dx;

    for(auto it = trainable_layers.rbegin(); it != trainable_layers.rend(); it++){
        (*it)->backward(temp);
        temp = (*it)->dl_dx;
    }
        
    return ;
}

float Loss_Base::get_loss(){
    return this->loss;
}