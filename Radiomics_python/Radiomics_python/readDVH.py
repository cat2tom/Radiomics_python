import dicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
import scipy.optimize as optimize

def getDoseParameters(sequence_num, rtss, rtdose):

    print "ROI Name: ", rtss.StructureSetROISequence[sequence_num].ROIName

    sequence = rtdose.DVHSequence[sequence_num]

    assert sequence.DVHType == 'CUMULATIVE'
    assert sequence.DVHVolumeUnits == 'CM3'
    assert sequence.DoseUnits == 'GY'

    print "Max Dose: ", round(sequence.DVHMaximumDose * 100, 2)
    print "Mean Dose: ", round(sequence.DVHMeanDose * 100, 2)
    print "Min Dose: ", round(sequence.DVHMinimumDose * 100, 2)

    dvh = sequence.DVHData

    dose_bin_width = np.array(map(float, dvh[::2])) * float(sequence.DVHDoseScaling)

    dose = []

    for i, bin_width in enumerate(dose_bin_width):

        if i == 0:

            dose.append(0)

        elif i > 0:

            dose.append(dose[-1] + bin_width)

    dose = np.array(dose)

    volume = np.array(map(float, dvh[1::2]))
    volume /= volume[0]

    linear = interp1d(dose, volume)

    def getD(x):

        root = optimize.brentq(lambda dose: linear(dose) - x, dose.min(), dose.max())

        return round(root * 100, 1)

    print "D99: ", getD(0.99)
    print "D95: ", getD(0.95)
    print "D90: ", getD(0.90)

    print linear(20.0)

rtss_path = r'rtss.dcm'
rtdose_path = r'rtdose.dcm'

rtss = dicom.read_file(rtss_path)
rtdose = dicom.read_file(rtdose_path)

for i, struct in enumerate(rtss.StructureSetROISequence):

    name = struct.ROIName

    print i, name

print "rtdose: ", rtdose

getDoseParameters(0, rtss, rtdose)
