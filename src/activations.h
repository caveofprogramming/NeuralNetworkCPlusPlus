#pragma once

#include <cmath>
#include "matrix.h"

namespace cop
{
    void softmax(float *pData, int nItems)
    {
        float sum = 0;
        float result = 0;
        float *pValue = pData;

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