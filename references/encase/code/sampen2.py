import math

from normalize_data import normalize_data


def sampen2(data, mm=2, r=0.2, normalize=False):
    """
    Calculates an estimate of sample entropy and the variance of the estimate.

    :param data: The data set (time series) as a list of floats.
    :type data: list

    :param mm: Maximum length of epoch (subseries).
    :type mm: int

    :param r: Tolerance. Typically 0.1 or 0.2.
    :type r: float

    :param normalize: Normalize such that the mean of the input is 0 and
    the sample, variance is 1.
    :type normalize: bool

    :return: List[(Int, Float/None, Float/None)...]

    Where the first (Int) value is the Epoch length.
    The second (Float or None) value is the SampEn.
    The third (Float or None) value is the Standard Deviation.

    The outputs are the sample entropies of the input, for all epoch lengths of
    0 to a specified maximum length, m.

    If there are no matches (the data set is unique) the sample entropy and
    standard deviation will return None.

    :rtype: list
    """

    n = len(data)

    if n == 0:
        raise ValueError("Parameter `data` contains an empty list")

    if mm > n / 2:
        raise ValueError(
            "Maximum epoch length of %d too large for time series of length "
            "%d (mm > n / 2)" % (
                mm,
                n,
            )
        )

    mm += 1

    mm_dbld = 2 * mm

    if mm_dbld > n:
        raise ValueError(
            "Maximum epoch length of %d too large for time series of length "
            "%d ((mm + 1) * 2 > n)" % (
                mm,
                n,
            )
        )

    if normalize is True:
        data = normalize_data(data)

    # initialize the lists
    run = [0] * n
    run1 = run[:]

    r1 = [0] * (n * mm_dbld)
    r2 = r1[:]
    f = r1[:]

    f1 = [0] * (n * mm)
    f2 = f1[:]

    k = [0] * ((mm + 1) * mm)

    a = [0] * mm
    b = a[:]
    p = a[:]
    v1 = a[:]
    v2 = a[:]
    s1 = a[:]
    n1 = a[:]
    n2 = a[:]

    for i in range(n - 1):
        nj = n - i - 1
        y1 = data[i]

        for jj in range(nj):
            j = jj + i + 1

            if data[j] - y1 < r and y1 - data[j] < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]

                for m in range(m1):
                    a[m] += 1
                    if j < n - 1:
                        b[m] += 1
                    f1[i + m * n] += 1
                    f[i + n * m] += 1
                    f[j + n * m] += 1

            else:
                run[jj] = 0

        for j in range(mm_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]

        if nj > mm_dbld - 1:
            for j in range(mm_dbld, nj):
                run1[j] = run[j]

    for i in range(1, mm_dbld):
        for j in range(i - 1):
            r2[i + n * j] = r1[i - j - 1 + n * j]
    for i in range(mm_dbld, n):
        for j in range(mm_dbld):
            r2[i + n * j] = r1[i - j - 1 + n * j]
    for i in range(n):
        for m in range(mm):
            ff = f[i + n * m]
            f2[i + n * m] = ff - f1[i + n * m]
            k[(mm + 1) * m] += ff * (ff - 1)
    m = mm - 1
    while m > 0:
        b[m] = b[m - 1]
        m -= 1
    b[0] = float(n) * (n - 1.0) / 2.0
    for m in range(mm):
        #### added
        if float(b[m]) == 0:
            p[m] = 0.0
            v2[m] = 0.0
        else:
            p[m] = float(a[m]) / float(b[m])
            v2[m] = p[m] * (1.0 - p[m]) / b[m]
    for m in range(mm):
        d2 = m + 1 if m + 1 < mm - 1 else mm - 1
        for d in range(d2):
            for i1 in range(d + 1, n):
                i2 = i1 - d - 1
                nm1 = f1[i1 + n * m]
                nm3 = f1[i2 + n * m]
                nm2 = f2[i1 + n * m]
                nm4 = f2[i2 + n * m]
                # if r1[i1 + n * j] >= m + 1:
                #    nm1 -= 1
                # if r2[i1 + n * j] >= m + 1:
                #    nm4 -= 1
                for j in range(2 * (d + 1)):
                    if r2[i1 + n * j] >= m + 1:
                        nm2 -= 1
                for j in range(2 * d + 1):
                    if r1[i2 + n * j] >= m + 1:
                        nm3 -= 1
                k[d + 1 + (mm + 1) * m] += float(2 * (nm1 + nm2) * (nm3 + nm4))

    n1[0] = float(n * (n - 1) * (n - 2))
    for m in range(mm - 1):
        for j in range(m + 2):
            n1[m + 1] += k[j + (mm + 1) * m]
    for m in range(mm):
        for j in range(m + 1):
            n2[m] += k[j + (mm + 1) * m]

    # calculate standard deviation for the set
    for m in range(mm):
        v1[m] = v2[m]
        ##### added
        if b[m] == 0:
            dv = 0.0
        else:
            dv = (n2[m] - n1[m] * p[m] * p[m]) / (b[m] * b[m])
        if dv > 0:
            v1[m] += dv
        s1[m] = math.sqrt(v1[m])

    # assemble and return the response
    response = []
    for m in range(mm):
        if p[m] == 0:
            # Infimum, the data set is unique, there were no matches.
            response.append((m, None, None))
        else:
            response.append((m, -math.log(p[m]), s1[m]))
    return response


if __name__ == '__main__':
#    print(sampen2(np.array([2.0, 1.0, 2.0, 2.0, 1.0, 2.0])))
    print(sampen2(np.array([1,1,1,1, 1, 1])))






