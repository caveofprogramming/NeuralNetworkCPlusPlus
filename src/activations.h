#pragma once

#include <cmath>
#include "matrix.h"

namespace cop
{
    void softmax(double *pData, int nItems)
    {
        double sum = 0;
        double result = 0;
        double *pValue = pData;

        for (int i = 0; i < nItems; i++)
        {
            result = exp(*pValue);
            sum += result;
            *pValue = result;

            ++pValue;
        }

        pValue = pData;

        for(int i = 0; i < nItems; i++)
        {
            *pValue /= sum;
            ++pValue;
        }
    }
}