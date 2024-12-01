#include "matrix.h"
#include "base.h"

class Optimizer_SGD : public Optimizer_Base{
    
    private:

    float momentum;

    void update_params(std::vector<Layer_Base*>& trainable_layers) const override;

    public:

    Optimizer_SGD(float learning_rate, float decay, float momentum);
    ~Optimizer_SGD() override;

    Optimizer_SGD(const Optimizer_SGD&) = delete;
    Optimizer_SGD& operator=(const Optimizer_SGD&) = delete;
    Optimizer_SGD(Optimizer_SGD&&) = delete;
    Optimizer_SGD& operator=(Optimizer_SGD&&) = delete;

};

class Optimizer_Adagrad : public Optimizer_Base{
    
    private:

    float epsilon;

    void update_params(std::vector<Layer_Base*>& trainable_layers) const override;

    public:

    Optimizer_Adagrad(float learning_rate, float decay, float epsilon);
    ~Optimizer_Adagrad() override;

    Optimizer_Adagrad(const Optimizer_Adagrad&) = delete;
    Optimizer_Adagrad& operator=(const Optimizer_Adagrad&) = delete;
    Optimizer_Adagrad(Optimizer_Adagrad&&) = delete;
    Optimizer_Adagrad& operator=(Optimizer_Adagrad&&) = delete;
    
};

class Optimizer_RMSprop : public Optimizer_Base{
    
    private:

    float epsilon;
    float rho;

    void update_params(std::vector<Layer_Base*>& trainable_layers) const override;

    public:

    Optimizer_RMSprop(float learning_rate, float decay, float epsilon, float rho);
    ~Optimizer_RMSprop() override;

    Optimizer_RMSprop(const Optimizer_RMSprop&) = delete;
    Optimizer_RMSprop& operator=(const Optimizer_RMSprop&) = delete;
    Optimizer_RMSprop(Optimizer_RMSprop&&) = delete;
    Optimizer_RMSprop& operator=(Optimizer_RMSprop&&) = delete;
    
};

class Optimizer_Adam : public Optimizer_Base{
    
    private:

    float epsilon;
    float beta1;
    float beta2;

    void update_params(std::vector<Layer_Base*>& trainable_layers) const override;

    float bin_pow(float a, int b) const;

    public:

    Optimizer_Adam(float learning_rate, float decay, float epsilon, float beta1, float beta2);
    ~Optimizer_Adam() override;

    Optimizer_Adam(const Optimizer_Adam&) = delete;
    Optimizer_Adam& operator=(const Optimizer_Adam&) = delete;
    Optimizer_Adam(Optimizer_Adam&&) = delete;
    Optimizer_Adam& operator=(Optimizer_Adam&&) = delete;
    
};