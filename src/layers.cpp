#include "layers.h"

Layer_Dense::Layer_Dense(int n_inputs, int n_neurons, float weight_l1, float weight_l2, float bias_l1, float bias_l2) : weight_regularizer_l1{weight_l1}, weight_regularizer_l2{weight_l2}, bias_regularizer_l1{bias_l1}, bias_regularizer_l2{bias_l2}{
    this->weights = Matrix<float>::random_matrix(n_inputs, n_neurons) * 0.01;
    this->bias = Matrix<float>::zero_matrix(1, n_neurons);
    this->weight_momentums = Matrix<float>::zero_matrix(n_inputs, n_neurons);
    this->bias_momentums = Matrix<float>::zero_matrix(1, n_neurons);
    this->weight_cache = Matrix<float>::zero_matrix(n_inputs, n_neurons);
    this->bias_cache = Matrix<float>::zero_matrix(1, n_neurons);
    this->dl_dx = Matrix<float>::zero_matrix(n_inputs, n_neurons);
    this->dl_dw = Matrix<float>::zero_matrix(n_inputs, n_neurons);
    this->dl_db = Matrix<float>::zero_matrix(n_inputs, n_neurons);
    this->require_grad = true;
}

Layer_Dense::~Layer_Dense() = default;

Matrix<float> Layer_Dense::forward(const Matrix<float>& inputs){
    this->inputs = inputs;
    this->output = this->inputs.mat_mul(this->weights) + this->bias;

    return this->output;
}

void Layer_Dense::backward(const Matrix<float>& dl_dz){
    this->dl_dw = this->inputs.transpose().mat_mul(dl_dz);
    this->dl_db = dl_dz.sum(1);
    this->dl_dx = dl_dz.mat_mul(this->weights.transpose());

    int r = this->weights.get_rows();
    int c = this->weights.get_cols();
        
    if(this->weight_regularizer_l1){
        Matrix<float> dL1(r, c, 0.0);
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                if(this->weights.get_val(i, j) < 0)
                    dL1.set_val(i, j, -1.0);
                else
                    dL1.set_val(i, j, 1.0);
            }
        }
        this->dl_dw = (this->dl_dw + (dL1 * this->weight_regularizer_l1));
    }

    if(this->weight_regularizer_l2)
        this->dl_dw = (this->dl_dw + (this->weights * (2 * this->weight_regularizer_l2)));
        
    if(this->bias_regularizer_l1 > 0){
        Matrix<float> dB1(1, c, 0.0);
        for(int j=0;j<c;j++){
            if(this->bias.get_val(0, j) < 0)
                dB1.set_val(0, j, -1.0);
            else
                dB1.set_val(0, j, 1.0);
        }
        this->dl_db = (this->dl_db + (dB1 * this->bias_regularizer_l1));
    }

    if(this->bias_regularizer_l2)
        this->dl_db = (this->dl_db + (this->bias * (2 * this->bias_regularizer_l2)));

    return ;
}