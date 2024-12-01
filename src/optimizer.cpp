#include "optimizer.h"

Optimizer_SGD::Optimizer_SGD(float learning_rate, float decay, float momentum) : Optimizer_Base{learning_rate, decay}, momentum{momentum}{}
Optimizer_SGD::~Optimizer_SGD() = default;

void Optimizer_SGD::update_params(std::vector<Layer_Base*>& trainable_layers) const{
    for(auto& layer:trainable_layers){
        if(!layer->require_grad)
            continue;

        Matrix<float> weight_updates = ((this->momentum * layer->weight_momentums) - (this->curr_learning_rate * layer->dl_dw));
        Matrix<float> bias_updates = ((this->momentum * layer->bias_momentums) - (this->curr_learning_rate * layer->dl_db));

        layer->weight_momentums = weight_updates;
        layer->bias_momentums = bias_updates;

        layer->weights = layer->weights + weight_updates;
        layer->bias = layer->bias + bias_updates;
    }

    return ;
}


Optimizer_Adagrad::Optimizer_Adagrad(float learning_rate, float decay, float epsilon) : Optimizer_Base{learning_rate, decay}, epsilon{epsilon}{}
Optimizer_Adagrad::~Optimizer_Adagrad() = default;

void Optimizer_Adagrad::update_params(std::vector<Layer_Base*>& trainable_layers) const{
    for(auto& layer:trainable_layers){
        if(!layer->require_grad)
            continue;

        layer->weight_cache = layer->weight_cache + layer->dl_dw.square();
        layer->bias_cache = layer->bias_cache + layer->dl_db.square();

        layer->weights = layer->weights - ((this->curr_learning_rate * layer->dl_dw) / (layer->weight_cache.sqrt() + this->epsilon));
        layer->bias = layer->bias - ((this->curr_learning_rate * layer->dl_db) / (layer->bias_cache.sqrt() + this->epsilon));
    }

    return ;
}


Optimizer_RMSprop::Optimizer_RMSprop(float learning_rate, float decay, float epsilon, float rho) : Optimizer_Base{learning_rate, decay}, epsilon{epsilon}, rho{rho}{}
Optimizer_RMSprop::~Optimizer_RMSprop() = default;

void Optimizer_RMSprop::update_params(std::vector<Layer_Base*>& trainable_layers) const{
    for(auto& layer:trainable_layers){
        if(!layer->require_grad)
            continue;

        layer->weight_cache = (this->rho * layer->weight_cache) + ((1 - this->rho) * layer->dl_dw.square());
        layer->bias_cache = (this->rho * layer->bias_cache) + ((1- this->rho) * layer->dl_db.square());

        layer->weights = layer->weights - ((this->curr_learning_rate * layer->dl_dw) / (layer->weight_cache.sqrt() + this->epsilon));
        layer->bias = layer->bias - ((this->curr_learning_rate * layer->dl_db) / (layer->bias_cache.sqrt() + this->epsilon));
    }

    return ;
}


Optimizer_Adam::Optimizer_Adam(float learning_rate, float decay, float epsilon, float beta1, float beta2) : Optimizer_Base{learning_rate, decay}, epsilon{epsilon}, beta1{beta1}, beta2{beta2}{}
Optimizer_Adam::~Optimizer_Adam() = default;

void Optimizer_Adam::update_params(std::vector<Layer_Base*>& trainable_layers) const{
    for(auto& layer:trainable_layers){
        if(!layer->require_grad)
            continue;

        layer->weight_momentums = (this->beta1 * layer->weight_momentums) + ((1 - this->beta1) * layer->dl_dw);
        layer->bias_momentums = (this->beta1 * layer->bias_momentums) + ((1 - this->beta1) * layer->dl_db);
        layer->weight_cache = (this->beta2 * layer->weight_cache) + ((1 - this->beta2) * layer->dl_dw.square());
        layer->bias_cache = (this->beta2 * layer->bias_cache) + ((1 - this->beta2) * layer->dl_db.square());

        Matrix<float> weight_momentums_corrected = layer->weight_momentums / (1 - (this->bin_pow(this->beta1, this->iterations+1)));
        Matrix<float> bias_momentums_corrected = layer->bias_momentums / (1 - (this->bin_pow(this->beta1, this->iterations+1)));
        Matrix<float> weight_cache_corrected = layer->weight_cache / (1 - (this->bin_pow(this->beta2, this->iterations+1)));
        Matrix<float> bias_cache_corrected = layer->bias_cache / (1 - (this->bin_pow(this->beta2, this->iterations+1)));

        layer->weights = layer->weights - ((this->curr_learning_rate * weight_momentums_corrected ) / (weight_cache_corrected.sqrt() + this->epsilon));
        layer->bias = layer->bias - ((this->curr_learning_rate * bias_cache_corrected) / (bias_cache_corrected.sqrt() + this->epsilon));
    }

    return ;
}

float Optimizer_Adam::bin_pow(float a, int b) const{
    float res = 1;
    while(b){
        if(b & 1)
            res *= a;
        a *= a;
        b >>= 1;
    }
    return res;
}