#pragma once
#include <vector>
#include <stdexcept>

template <typename T>
class Matrix{
    private:
    int m_rows;
    int m_cols;
    std::vector<std::vector<T>> m_arr;

    public:

    Matrix();

    Matrix(int rows, int cols, T ini = T{});

    ~Matrix();

    Matrix(const Matrix<T>& a);

    Matrix<T>& operator=(const Matrix<T>& a);

    Matrix(Matrix<T>&& a) noexcept;

    Matrix<T>& operator=(Matrix<T>&& a) noexcept;

    int get_rows() const;

    int get_cols() const;

    T get_val(int i, int j) const;

    void set_val(int i, int j, T data);

    static Matrix<T> zero_matrix(int rows, int cols);

    static Matrix<T> random_matrix(int rows, int cols);

    Matrix<T> transpose() const;

    Matrix<T> operator+(const Matrix<T>& other) const;

    Matrix<T> operator+(const T scalar) const;

    template <typename U>
    friend Matrix<U> operator+(const U scalar, const Matrix<U>& other);

    Matrix<T> operator-() const;

    Matrix<T> operator-(const Matrix<T>& other) const;

    Matrix<T> operator-(const T scalar) const;

    template <typename U>
    friend Matrix<U> operator-(const U scalar, const Matrix<U>& other);

    Matrix<T> operator*(const Matrix<T>& other) const;

    Matrix<T> operator*(const T scalar) const;

    template <typename U>
    friend Matrix<U> operator*(const U scalar, const Matrix<U>& other);

    Matrix<T> operator/(const Matrix<T>& other) const;

    Matrix<T> operator/(const T scalar) const;

    template <typename U>
    friend Matrix<U> operator/(const U scalar, const Matrix<U>& other);

    Matrix<T> mat_mul(const Matrix<T>& other) const;

    Matrix<T> sum(int axis = -1) const;

    Matrix<T> square() const;

    Matrix<T> abs() const;

    Matrix<T> sqrt() const;

    Matrix<T> log(float base = 2.0) const;

    Matrix<T> clip(T low, T high) const;

    void set_zero();

    void print() const;

};

template <typename U>
Matrix<U> operator+(const U scalar, const Matrix<U>& other){
    Matrix<U> temp (other.m_rows, other.m_cols, 0);

    for(int i=0;i<other.m_rows;i++){
        for(int j=0;j<other.m_cols;j++)
            temp.m_arr[i][j] = other.m_arr[i][j] + scalar;
    }

    return temp;
}

template <typename U>
Matrix<U> operator-(const U scalar, const Matrix<U>& other){
    Matrix<U> temp(other.m_rows, other.m_cols, 0);

    for(int i=0;i<temp.m_rows;i++){
        for(int j=0;j<temp.m_cols;j++)
            temp.m_arr[i][j] = scalar - other.m_arr[i][j];
    }

    return temp;
}

template <typename U>
Matrix<U> operator*(const U scalar, const Matrix<U>& other){
    Matrix<U> temp(other.m_rows, other.m_cols, 0);

    for(int i=0;i<other.m_rows;i++){
        for(int j=0;j<other.m_cols;j++)
            temp.m_arr[i][j] = other.m_arr[i][j] * scalar;
    }

    return temp;
}

template <typename U>
Matrix<U> operator/(const U scalar, const Matrix<U>& other){
    Matrix<U> temp(other.m_rows, other.m_cols, 0);

    for(int i=0;i<temp.m_rows;i++){
        for(int j=0;j<temp.m_cols;j++){
            if (other.m_arr[i][j] == 0)
                throw std::invalid_argument("Division by zero is not allowed!!!");
            temp.m_arr[i][j] = scalar / other.m_arr[i][j];
        }
    }

    return temp;
}