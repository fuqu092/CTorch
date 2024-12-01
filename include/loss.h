#pragma once

#include "matrix.h"
#include "base.h"

// In case of loss functions dl_dz == y_pred

class Loss_Categorical_Cross_Entropy : public Loss_Base{

    public:

    Loss_Categorical_Cross_Entropy();
    ~Loss_Categorical_Cross_Entropy() override;

    Loss_Categorical_Cross_Entropy(const Loss_Categorical_Cross_Entropy&) = delete;
    Loss_Categorical_Cross_Entropy& operator=(const Loss_Categorical_Cross_Entropy&) = delete;
    Loss_Categorical_Cross_Entropy(Loss_Categorical_Cross_Entropy&&) = delete;
    Loss_Categorical_Cross_Entropy& operator=(Loss_Categorical_Cross_Entropy&&) = delete;

    // y_true is assumed to be a batches x 1 vector or batches x classes matrix
    float forward(const Matrix<float>& y_pred, const Matrix<float>& y_true) const override;
    void backward(const Matrix<float>& y_pred, const Matrix<float>& y_true) override;

};

class Loss_Binary_Cross_Entropy : public Loss_Base{

    public:

    Loss_Binary_Cross_Entropy();
    ~Loss_Binary_Cross_Entropy() override;

    Loss_Binary_Cross_Entropy(const Loss_Binary_Cross_Entropy&) = delete;
    Loss_Binary_Cross_Entropy& operator=(const Loss_Binary_Cross_Entropy&) = delete;
    Loss_Binary_Cross_Entropy(Loss_Binary_Cross_Entropy&&) = delete;
    Loss_Binary_Cross_Entropy& operator=(Loss_Binary_Cross_Entropy&&) = delete;

    // y_true is assumed to be a batches x 1 vector or batches x classes matrix
    float forward(const Matrix<float>& y_pred, const Matrix<float>& y_true) const override;
    void backward(const Matrix<float>& y_pred, const Matrix<float>& y_true) override;

};

class Loss_Mean_Squared_Error : public Loss_Base{

    public:

    Loss_Mean_Squared_Error();
    ~Loss_Mean_Squared_Error() override;

    Loss_Mean_Squared_Error(const Loss_Mean_Squared_Error&) = delete;
    Loss_Mean_Squared_Error& operator=(const Loss_Mean_Squared_Error&) = delete;
    Loss_Mean_Squared_Error(Loss_Mean_Squared_Error&&) = delete;
    Loss_Mean_Squared_Error& operator=(Loss_Mean_Squared_Error&&) = delete;

    float forward(const Matrix<float>& y_pred, const Matrix<float>& y_true) const override;
    void backward(const Matrix<float>& y_pred, const Matrix<float>& y_true) override;

};

class Loss_Mean_Absolute_Error : public Loss_Base{

    public:

    Loss_Mean_Absolute_Error();
    ~Loss_Mean_Absolute_Error() override;

    Loss_Mean_Absolute_Error(const Loss_Mean_Absolute_Error&) = delete;
    Loss_Mean_Absolute_Error& operator=(const Loss_Mean_Absolute_Error&) = delete;
    Loss_Mean_Absolute_Error(Loss_Mean_Absolute_Error&&) = delete;
    Loss_Mean_Absolute_Error& operator=(Loss_Mean_Absolute_Error&&) = delete;

    float forward(const Matrix<float>& y_pred, const Matrix<float>& y_true) const override;
    void backward(const Matrix<float>& y_pred, const Matrix<float>& y_true) override;

};