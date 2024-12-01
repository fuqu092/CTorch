#include <iostream>
#include <stdexcept>
#include <random>
#include "matrix.h"

template <typename T>
Matrix<T>::Matrix(): m_rows{0}, m_cols{0} {}

template <typename T>
Matrix<T>::Matrix(int rows, int cols, T ini) : m_rows{rows}, m_cols{cols}, m_arr{std::vector<std::vector<T>> (rows, std::vector<T> (cols, ini))}{}

template <typename T>
Matrix<T>::~Matrix() = default;

template <typename T>
Matrix<T>::Matrix(const Matrix<T>& a): m_rows{a.m_rows}, m_cols{a.m_cols}, m_arr{a.m_arr}{}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& a){
    if(this == &a)
        return *this;

    this->m_rows = a.m_rows;
    this->m_cols = a.m_cols;
    this->m_arr = a.m_arr;

    return *this;
}

template <typename T>
Matrix<T>::Matrix(Matrix<T>&& a) noexcept : m_rows{a.m_rows}, m_cols{a.m_cols}{
    this->m_arr = std::move(a.m_arr);
    a.m_rows = 0;
    a.m_cols = 0;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& a) noexcept{
    if(this == &a)
        return *this;

    this->m_arr = std::move(a.m_arr);
    this->m_rows = a.m_rows;
    this->m_cols = a.m_cols;

    a.m_rows = 0;
    a.m_cols = 0;

    return *this;
}

template <typename T>
int Matrix<T>::get_rows() const{
    return this->m_rows;
}

template <typename T>
int Matrix<T>::get_cols() const{
    return this->m_cols;
}

template <typename T>
T Matrix<T>::get_val(int i, int j) const{
    if(i >= 0 && i < this->m_rows && j >= 0 && j < this->m_cols)
        return this->m_arr[i][j];
    else
        throw std::invalid_argument("Indices out of range!!!");
}

template <typename T>
void Matrix<T>::set_val(int i, int j, T data){
    if(i >= 0 && i < this->m_rows && j >= 0 && j < this->m_cols)
        this->m_arr[i][j] = data;
    else
        throw std::invalid_argument("Indices out of range!!!");
    return ;
}

template <typename T>
Matrix<T> Matrix<T>::zero_matrix(int rows, int cols){
    Matrix<T> temp(rows, cols, T(0));
    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::random_matrix(int rows, int cols){
    std::random_device rd;
    std::mt19937 gen(rd());

    Matrix<T> temp(rows, cols, 0);

    for(int i=0;i<rows;i++){
        std::normal_distribution<float> dist(0.0, 1.0);
        for(int j=0;j<cols;j++)
            temp.m_arr[i][j] = dist(gen);
    }
        
    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const{
    Matrix<T> temp(this->m_cols, this->m_rows, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            temp.m_arr[j][i] = this->m_arr[i][j];
    }
        
    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const{
    if(this->m_rows == other.m_rows && this->m_cols == other.m_cols){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);
            
        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][j] + other.m_arr[i][j];
        }

        return temp;
    }

    else if(other.m_rows == 1 && this->m_cols == other.m_cols){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][j] + other.m_arr[0][j];
        }

        return temp;
    }

    else if(other.m_cols == 1 && this->m_rows == other.m_rows){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][j] + other.m_arr[i][0];
        }

        return temp;
    }

    else if(this->m_rows == 1 && this->m_cols == other.m_cols){
        Matrix<T> temp(other.m_rows, other.m_cols, 0);

        for(int i=0;i<other.m_rows;i++){
            for(int j=0;j<other.m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[0][j] + other.m_arr[i][j];
        }

        return temp;
    }

    else if(this->m_cols == 1 && this->m_rows == other.m_rows){
        Matrix<T> temp(other.m_rows, other.m_cols, 0);

        for(int i=0;i<other.m_rows;i++){
            for(int j=0;j<other.m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][0] + other.m_arr[i][j];
        }

        return temp;
    }

    throw std::invalid_argument("Invalid dimensions for addition!!!");
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const T scalar) const{
    Matrix<T> temp(this->m_rows, this->m_cols, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            temp.m_arr[i][j] = this->m_arr[i][j] + scalar;
    }

    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::operator-() const{
    Matrix<T> temp(this->m_rows, this->m_cols, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            temp.m_arr[i][j] = -this->m_arr[i][j];
    }

    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const{
    if(this->m_rows == other.m_rows && this->m_cols == other.m_cols){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][j] - other.m_arr[i][j];
        }

        return temp;
    }

    else if(other.m_rows == 1 && this->m_cols == other.m_cols){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][j] - other.m_arr[0][j];
        }

        return temp;
    }

    else if(other.m_cols == 1 && this->m_rows == other.m_rows){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][j] - other.m_arr[i][0];
        }

        return temp;
    }

    else if(this->m_rows == 1 && this->m_cols == other.m_cols){
        Matrix<T> temp(other.m_rows, other.m_cols, 0);

        for(int i=0;i<other.m_rows;i++){
            for(int j=0;j<other.m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[0][j] - other.m_arr[i][j];
        }

        return temp;
    }

    else if(this->m_cols == 1 && this->m_rows == other.m_rows){
        Matrix<T> temp(other.m_rows, other.m_cols, 0);

        for(int i=0;i<other.m_rows;i++){
            for(int j=0;j<other.m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][0] - other.m_arr[i][j];
        }

        return temp;
    }

    throw std::invalid_argument("Invalid dimensions for subtraction!!!");
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const T scalar) const{
    Matrix<T> temp(this->m_rows, this->m_cols, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            temp.m_arr[i][j] = this->m_arr[i][j] - scalar;
    }

    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const{
    if(this->m_rows == other.m_rows && this->m_cols == other.m_cols){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][j] * other.m_arr[i][j];
        }

        return temp;
    }

    else if(other.m_rows == 1 && this->m_cols == other.m_cols){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][j] * other.m_arr[0][j];
        }

        return temp;
    }

    else if(other.m_cols == 1 && this->m_rows == other.m_rows){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][j] * other.m_arr[i][0];
        }

        return temp;
    }

    else if(this->m_rows == 1 && this->m_cols == other.m_cols){
        Matrix<T> temp(other.m_rows, other.m_cols, 0);

        for(int i=0;i<other.m_rows;i++){
            for(int j=0;j<other.m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[0][j] * other.m_arr[i][j];
        }

        return temp;
    }

    else if(this->m_cols == 1 && this->m_rows == other.m_rows){
        Matrix<T> temp(other.m_rows, other.m_cols, 0);

        for(int i=0;i<other.m_rows;i++){
            for(int j=0;j<other.m_cols;j++)
                temp.m_arr[i][j] = this->m_arr[i][0] * other.m_arr[i][j];
        }

        return temp;
    }

    throw std::invalid_argument("Invalid dimensions for element-wise multiplication!!!");
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T scalar) const{
    Matrix<T> temp(this->m_rows, this->m_cols, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            temp.m_arr[i][j] = this->m_arr[i][j] * scalar;
    }

    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::operator/(const Matrix<T>& other) const{
    if(this->m_rows == other.m_rows && this->m_cols == other.m_cols){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++){
                if (other.m_arr[i][j] == 0)
                    throw std::invalid_argument("Division by zero is not allowed!!!");
                temp.m_arr[i][j] = this->m_arr[i][j] / other.m_arr[i][j];
            }
        }

        return temp;
    }

    else if(other.m_rows == 1 && this->m_cols == other.m_cols){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++){
                if(other.m_arr[i][j] == 0)
                    throw std::invalid_argument("Division by zero is not allowed!!!");
                temp.m_arr[i][j] = this->m_arr[i][j] / other.m_arr[0][j];
            }
        }

        return temp;
    }

    else if(other.m_cols == 1 && this->m_rows == other.m_rows){
        Matrix<T> temp(this->m_rows, this->m_cols, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++){
                if(other.m_arr[i][j] == 0)
                    throw std::invalid_argument("Division by zero is not allowed!!!");
                temp.m_arr[i][j] = this->m_arr[i][j] / other.m_arr[i][0];
            }
        }

        return temp;
    }

    else if(this->m_rows == 1 && this->m_cols == other.m_cols){
        Matrix<T> temp(other.m_rows, other.m_cols, 0);

        for(int i=0;i<other.m_rows;i++){
            for(int j=0;j<other.m_cols;j++){
                if (other.m_arr[i][j] == 0)
                    throw std::invalid_argument("Division by zero is not allowed!!!");
                temp.m_arr[i][j] = this->m_arr[0][j] / other.m_arr[i][j];
            }
        }

        return temp;
    }

    else if(this->m_cols == 1 && this->m_rows == other.m_rows){
        Matrix<T> temp(other.m_rows, other.m_cols, 0);

        for(int i=0;i<other.m_rows;i++){
            for(int j=0;j<other.m_cols;j++){
                if (other.m_arr[i][j] == 0)
                    throw std::invalid_argument("Division by zero is not allowed!!!");
                temp.m_arr[i][j] = this->m_arr[i][0] / other.m_arr[i][j];
            }
        }

        return temp;
    }

    throw std::invalid_argument("Invalid dimensions for element-wise divison!!!");
}

template <typename T>
Matrix<T> Matrix<T>::operator/(const T scalar) const{
    if(scalar == 0)
        throw std::invalid_argument("Divison by zero is not allowed!!!");

    Matrix<T> temp(this->m_rows, this->m_cols, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            temp.m_arr[i][j] = this->m_arr[i][j] / scalar;
    }

    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::mat_mul(const Matrix<T>& other) const{
    if(this->m_cols != other.m_rows)
        throw std::invalid_argument("Dimensions don't match for matrix multiplication!!!");

    Matrix<T> temp(this->m_rows, other.m_cols, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int k=0;k<this->m_cols;k++){
            for(int j=0;j<other.m_cols;j++)
                temp.m_arr[i][j] += this->m_arr[i][k] * other.m_arr[k][j];
        }
    }

    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::sum(int axis) const{
    if(axis == -1) { // Total sum
        Matrix<T> temp(1, 1, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++) {
                temp.m_arr[0][0] += this->m_arr[i][j];
            }
        }

        return temp;
    }

    else if(axis == 0) { // Sum across rows
        Matrix<T> temp(this->m_rows, 1, 0);

        for(int i=0;i<this->m_rows;i++){
            for(int j=0;j<this->m_cols;j++)
                temp.m_arr[i][0] += this->m_arr[i][j];
        }

        return temp;
    }

    else if(axis == 1){ // Sum across columns
        Matrix<T> temp(1, this->m_cols, 0);

        for(int j=0; j<this->m_cols;j++){
            for(int i=0;i<this->m_rows;i++)
                temp.m_arr[0][j] += this->m_arr[i][j];
        }

        return temp;
    }

    throw std::invalid_argument("Invalid axis to sum!!!");
}

template <typename T>
Matrix<T> Matrix<T>::square() const{
    Matrix<T> temp(this->m_rows, this->m_cols, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            temp.m_arr[i][j] = this->m_arr[i][j] * this->m_arr[i][j];
    }

    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::abs() const{
    Matrix<T> temp(this->m_rows, this->m_cols, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            temp.m_arr[i][j] = std::abs(this->m_arr[i][j]);
    }

    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::sqrt() const{
    Matrix<T> temp(this->m_rows, this->m_cols, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            temp.m_arr[i][j] = std::sqrt(this->m_arr[i][j]);
    }

    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::log(float base) const{
    Matrix<T> temp(this->m_rows, this->m_cols, 0);

    float base_log = std::log(base);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            temp.m_arr[i][j] = std::log(this->m_arr[i][j]) / base_log;
    }

    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::clip(T low, T high) const{
    Matrix<T> temp(this->m_rows, this->m_cols, 0);

    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++){
            if(this->m_arr[i][j] < low)
                temp.m_arr[i][j] = low;
            else if(this->m_arr[i][j] > high)
                temp.m_arr[i][j] = high;
            else
                temp.m_arr[i][j] = this->m_arr[i][j];
        }
    }

    return temp;
}

template <typename T>
void Matrix<T>::set_zero() {
    for(int i=0;i<this->m_rows;i++) {
        for(int j=0;j<this->m_cols;j++) {
            this->m_arr[i][j] = T(0);
        }
    }

    return ;
}

template <typename T>
void Matrix<T>::print() const{
    for(int i=0;i<this->m_rows;i++){
        for(int j=0;j<this->m_cols;j++)
            std::cout<<this->m_arr[i][j]<<" ";
        std::cout<<'\n';
    }
    return ;
}

template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<long long>;