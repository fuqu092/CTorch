#include "activation.h"


Activation_ReLU::Activation_ReLU(){
    this->require_grad = false;
}
Activation_ReLU::~Activation_ReLU() = default;

Matrix<float> Activation_ReLU::forward(const Matrix<float>& inputs){
    this->inputs = inputs;
    this->output = inputs;

    for(int i=0;i<this->output.get_rows();i++){
        for(int j=0;j<this->output.get_cols();j++){
            if(this->output.get_val(i, j) < 0)
                this->output.set_val(i, j, 0);
        }
    }
    return this->output;
}

void Activation_ReLU::backward(const Matrix<float>& dl_dz){
    this->dl_dx = dl_dz;

    for(int i=0;i<this->dl_dx.get_rows();i++){
        for(int j=0;j<this->dl_dx.get_cols();j++){
            if(this->inputs.get_val(i, j) < 0)
                this->dl_dx.set_val(i, j, 0);
        }
    }
    return ;
}