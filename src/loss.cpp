#include "loss.h"

// In case of loss functions dl_dz == y_pred

Loss_Categorical_Cross_Entropy::Loss_Categorical_Cross_Entropy() = default;
Loss_Categorical_Cross_Entropy::~Loss_Categorical_Cross_Entropy() = default;

// y_true is assumed to be a batches x 1 vector or batches x classes matrix
float Loss_Categorical_Cross_Entropy::forward(const Matrix<float>& y_pred, const Matrix<float>& y_true) const{
    int samples = y_pred.get_rows();
    Matrix<float> y_pred_clipped = y_pred.clip(1e-7, 1.0-1e-7);

    Matrix<float> correct_confidences(samples, 1);
    if(y_true.get_cols() == 1){
        for(int i=0;i<samples;i++)
            correct_confidences.set_val(i, 0, y_pred_clipped.get_val(i, y_true.get_val(i, 0)));
    }
    else
        correct_confidences = (y_pred_clipped * y_true).sum(0);

    float loss = ((-correct_confidences.log()).sum(-1).get_val(0, 0)) / (samples * 1.0);
    return loss;

}

void Loss_Categorical_Cross_Entropy::backward(const Matrix<float>& y_pred, const Matrix<float>& y_true) {
    int samples = y_pred.get_rows();
    Matrix<float> y_pred_clipped = y_pred.clip(1e-7, 1.0 - 1e-7);
    Matrix<float> y_true_temp = y_true;

    if(y_true.get_cols() == 1){
        Matrix<float> temp = Matrix<float>::zero_matrix(samples, y_pred.get_cols());
        for(int i=0;i<samples;i++)
            temp.set_val(i, y_true.get_val(i, 0), 1.0);
        y_true_temp = temp;
    }

    this->dl_dx = -y_true_temp/(y_pred_clipped * samples);

    return ;
}


Loss_Binary_Cross_Entropy::Loss_Binary_Cross_Entropy() = default;
Loss_Binary_Cross_Entropy::~Loss_Binary_Cross_Entropy() = default;

// y_true is assumed to be a batches x 1 vector or batches x classes matrix
float Loss_Binary_Cross_Entropy::forward(const Matrix<float>& y_pred, const Matrix<float>& y_true) const{
    int samples = y_pred.get_rows();
    Matrix<float> y_pred_clipped = y_pred.clip(1e-7, 1.0-1e-7);

    float loss = -((y_true * y_pred_clipped.log()) + ((-y_true + 1) * ((-y_pred_clipped + 1).log()))).sum(-1).get_val(0, 0) / (samples * 1.0);
    return loss;
}

void Loss_Binary_Cross_Entropy::backward(const Matrix<float>& y_pred, const Matrix<float>& y_true){
    int samples = y_pred.get_rows();
    Matrix<float> y_pred_clipped = y_pred.clip(1e-7, 1.0 - 1e-7);

    this->dl_dx = -((y_true / y_pred_clipped) - ((-y_true + 1) / (-y_pred_clipped + 1))) / samples;
    return ;
}


Loss_Mean_Squared_Error::Loss_Mean_Squared_Error() = default;
Loss_Mean_Squared_Error::~Loss_Mean_Squared_Error() = default;

float Loss_Mean_Squared_Error::forward(const Matrix<float>& y_pred, const Matrix<float>& y_true) const{
    int samples = y_pred.get_rows();
    float loss = (((y_pred - y_true).square()).sum(-1)).get_val(0, 0) / (samples * 1.0);

    return loss;
}

void Loss_Mean_Squared_Error::backward(const Matrix<float>& y_pred, const Matrix<float>& y_true){
    int samples = y_pred.get_rows();

    this->dl_dx = ((y_pred - y_true) * 2)/ (samples * 1.0);

    return ;
}


Loss_Mean_Absolute_Error::Loss_Mean_Absolute_Error() = default;
Loss_Mean_Absolute_Error::~Loss_Mean_Absolute_Error() = default;

float Loss_Mean_Absolute_Error::forward(const Matrix<float>& y_pred, const Matrix<float>& y_true) const{
    int samples = y_pred.get_rows();
    float loss = (((y_pred - y_true).abs()).sum(-1)).get_val(0, 0) / (samples * 1.0);

    return loss;
}

void Loss_Mean_Absolute_Error::backward(const Matrix<float>& y_pred, const Matrix<float>& y_true){
    int samples = y_pred.get_rows();
        
    this->dl_dx = y_pred - y_true;
    for(int i=0;i<this->dl_dx.get_rows();i++){
        for(int j=0;j<this->dl_dx.get_cols();j++){
            if(this->dl_dx.get_val(i, j) < 0)
                this->dl_dx.set_val(i, j, -1.0/samples);
            else
                this->dl_dx.set_val(i, j, 1.0/samples);
        }
    }

    return ;
}