
/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*   
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include "formula.h"
#include "Param.h"
#include "Array.h"
#include "Mapping.h"
#include "NeuroSim.h"

extern Param *param;

extern std::vector< std::vector<double> > Input;
extern std::vector< std::vector<int> > dInput;
extern std::vector< std::vector<double> > Output;

extern std::vector< std::vector<double> > weight1;
extern std::vector< std::vector<double> > weight2;
extern std::vector< std::vector<double> > deltaWeight1;
extern std::vector< std::vector<double> > deltaWeight2;

extern Technology techIH;
extern Technology techHO;
extern Array *arrayIH;
extern Array *arrayHO;
extern SubArray *subArrayIH;
extern SubArray *subArrayHO;
extern Adder adderIH;
extern Mux muxIH;
extern RowDecoder muxDecoderIH;
extern DFF dffIH;
extern Adder adderHO;
extern Mux muxHO;
extern RowDecoder muxDecoderHO;
extern DFF dffHO;

void Train(const int numTrain, const int epochs) {
	int numBatchReadSynapse;	// # of read synapses in a batch read operation (decide later)
	int numBatchWriteSynapse;	// # of write synapses in a batch write operation (decide later)
	double outN1[param->nHide]; // Net input to the hidden layer [param->nHide]
	double a1[param->nHide];    // Net output of hidden layer [param->nHide] also the input of hidden layer to output layer
	int da1[param->nHide];  // Digitized net output of hidden layer [param->nHide] also the input of hidden layer to output layer
	double outN2[param->nOutput];   // Net input to the output layer [param->nOutput]
	double a2[param->nOutput];  // Net output of output layer [param->nOutput]
	//std::vector<std::vector<double>>
	double s1[param->nHide];    // Output delta from input layer to the hidden layer [param->nHide]
	double s2[param->nOutput];  // Output delta from hidden layer to the output layer [param->nOutput]
	for (int t = 0; t < epochs; t++) {
		for (int batchSize = 0; batchSize < numTrain; batchSize++) {

			int i = rand() % param->numMnistTrainImages;  // Randomize sample

			// Forward propagation
			/* First layer (input layer to the hidden layer) */
			std::fill_n(outN1, param->nHide, 0);
			std::fill_n(a1, param->nHide, 0);
			if (param->useHardwareInTrainingFF) {   // Hardware
				double sumArrayReadEnergy = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double readVoltage = static_cast<eNVM*>(arrayIH->cell[0][0])->readVoltage;
				double readPulseWidth = static_cast<eNVM*>(arrayIH->cell[0][0])->readPulseWidth;
#pragma omp parallel for reduction(+: sumArrayReadEnergy)
				for (int j = 0; j < param->nHide; j++) {
					if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
						if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
							sumArrayReadEnergy += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd * param->nInput; // All WLs open
						}
					}
					else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH->cell[0][0])) { // Digital eNVM
						if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
							sumArrayReadEnergy += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd; // Selected WL
						}
						else {    // Cross-point
							sumArrayReadEnergy += arrayIH->wireCapRow * techIH.vdd * techIH.vdd * (param->nInput - 1);  // Unselected WLs
						}
					}
					for (int n = 0; n < param->numBitInput; n++) {
						double pSumMaxAlgorithm = pow(2, n) / (param->numInputLevel - 1) * arrayIH->arrayRowSize;  // Max algorithm partial weighted sum for the nth vector bit (if both max input value and max weight are 1)
						//numInputlevel=2, pSumMaxAlgoritmh=100; 
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
							double Isum = 0;    // weighted sum current
							double IsumMax = 0; // Max weighted sum current
							double inputSum = 0;    // Weighted sum current of input vector * weight=1 column
							for (int k = 0; k < param->nInput; k++) {
								if ((dInput[i][k] >> n) & 1) {    // if the nth bit of dInput[i][k] is 1
									Isum += arrayIH->ReadCell(j, k);
									
									inputSum += arrayIH->GetMaxCellReadCurrent(j, k);
									sumArrayReadEnergy += arrayIH->wireCapRow * readVoltage * readVoltage; // Selected BLs (1T1R) or Selected WLs (cross-point)
								}
								IsumMax += arrayIH->GetMaxCellReadCurrent(j, k);
							}
							//std::cout << Isum << std::endl;
							//std::cout << IsumMax << std::endl;
							sumArrayReadEnergy += Isum * readVoltage * readPulseWidth;
							//int outputDigits = CurrentToDigits(Isum, IsumMax);
							int outputDigits = 2* CurrentToDigits(Isum, IsumMax) - CurrentToDigits(inputSum, IsumMax);
							//int outputDigits = 2*CurrentToDigits(Isum, IsumMax) - CurrentToDigits(inputSum, IsumMax);
							//std::cout << outputDigits << std::endl;
							outN1[j] += DigitsToAlgorithm(outputDigits, pSumMaxAlgorithm);
							//std::cout << outN1[j] << std::endl;
							//이값이 -10 ~ 10 사이값으로 나오야함.
						}
						else {    // SRAM or digital eNVM
							int Dsum = 0;
							int DsumMax = 0;
							int inputSum = 0;
							for (int k = 0; k < param->nInput; k++) {
								if ((dInput[i][k] >> n) & 1) {    // if the nth bit of dInput[i][k] is 1
									Dsum += (int)(arrayIH->ReadCell(j, k));
									inputSum += pow(2, arrayIH->numCellPerSynapse) - 1;
								}
								DsumMax += pow(2, arrayIH->numCellPerSynapse) - 1;
							}
							if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH->cell[0][0])) {    // Digital eNVM
								sumArrayReadEnergy += static_cast<DigitalNVM*>(arrayIH->cell[0][0])->readEnergy * arrayIH->numCellPerSynapse * arrayIH->arrayRowSize;
							}
							else {    // SRAM
								sumArrayReadEnergy += static_cast<SRAM*>(arrayIH->cell[0][0])->readEnergy * arrayIH->numCellPerSynapse * arrayIH->arrayRowSize;
							}
							outN1[j] += (double)(2 * Dsum - inputSum) / DsumMax * pSumMaxAlgorithm;
						}
					}
					a1[j] = sigmoid(outN1[j]);
					da1[j] = round_th(a1[j] * (param->numInputLevel - 1), param->Hthreshold);
					// -4.3 ==> -4로 출력 -4.7 ==> -5로 출력, 4.3 ==> 4 , 4.7 ==> 5
				}
				arrayIH->readEnergy += sumArrayReadEnergy;

				numBatchReadSynapse = (int)ceil((double)param->nHide / param->numColMuxed);
				// Don't parallelize this loop since there may be update of member variables inside NeuroSim functions
				for (int j = 0; j < param->nHide; j += numBatchReadSynapse) {
					int numActiveRows = 0;  // Number of selected rows for NeuroSim
					for (int n = 0; n < param->numBitInput; n++) {
						for (int k = 0; k < param->nInput; k++) {
							if ((dInput[i][k] >> n) & 1) {    // if the nth bit of dInput[i][k] is 1
								numActiveRows++;
							}
						}
					}
					subArrayIH->activityRowRead = (double)numActiveRows / param->nInput / param->numBitInput;
					subArrayIH->readDynamicEnergy += NeuroSimSubArrayReadEnergy(subArrayIH);
					subArrayIH->readDynamicEnergy += NeuroSimNeuronReadEnergy(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
					subArrayIH->readLatency += NeuroSimSubArrayReadLatency(subArrayIH);
					subArrayIH->readLatency += NeuroSimNeuronReadLatency(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
				}
			}
			else {    // Algorithm
#pragma omp parallel for
				for (int j = 0; j < param->nHide; j++) {
					for (int k = 0; k < param->nInput; k++) {
						outN1[j] += 2 * Input[i][k] * weight1[j][k] - Input[i][k];
					}
					a1[j] = sigmoid(outN1[j]);
				}
			}

			/* Second layer (hidder layer to the output layer) */
			std::fill_n(outN2, param->nOutput, 0);
			std::fill_n(a2, param->nOutput, 0);
			if (param->useHardwareInTrainingFF) {   // Hardware
				double sumArrayReadEnergy = 0;  // Use a temporary variable here since OpenMP does not support reduction on class member
				double readVoltage = static_cast<eNVM*>(arrayHO->cell[0][0])->readVoltage;
				double readPulseWidth = static_cast<eNVM*>(arrayHO->cell[0][0])->readPulseWidth;
#pragma omp parallel for reduction(+: sumArrayReadEnergy)
				for (int j = 0; j < param->nOutput; j++) {
					if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
						if (static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) {  // 1T1R
							sumArrayReadEnergy += arrayHO->wireGateCapRow * techHO.vdd * techHO.vdd * param->nHide; // All WLs open
						}
					}
					else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayHO->cell[0][0])) { // Digital eNVM
						if (static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) {  // 1T1R
							sumArrayReadEnergy += arrayHO->wireGateCapRow * techHO.vdd * techHO.vdd;    // Selected WL
						}
						else {    // Cross-point
							sumArrayReadEnergy += arrayHO->wireCapRow * techHO.vdd * techHO.vdd * (param->nHide - 1);   // Unselected WLs
						}
					}
					for (int n = 0; n < param->numBitInput; n++) {
						double pSumMaxAlgorithm = pow(2, n) / (param->numInputLevel - 1) * arrayHO->arrayRowSize;    // Max algorithm partial weighted sum for the nth vector bit (if both max input value and max weight are 1)
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
							double Isum = 0;    // weighted sum current
							double IsumMax = 0; // Max weighted sum current
							double a1Sum = 0;   // Weighted sum current of a1 vector * weight=1 column
							for (int k = 0; k < param->nHide; k++) {
								if ((da1[k] >> n) & 1) {    // if the nth bit of da1[k] is 1
									Isum += arrayHO->ReadCell(j, k);
									a1Sum += arrayHO->GetMaxCellReadCurrent(j, k);
									sumArrayReadEnergy += arrayHO->wireCapRow * readVoltage * readVoltage; // Selected BLs (1T1R) or Selected WLs (cross-point)
								}
								IsumMax += arrayHO->GetMaxCellReadCurrent(j, k);
							}
							sumArrayReadEnergy += Isum * readVoltage * readPulseWidth;
							int outputDigits = 2 * CurrentToDigits(Isum, IsumMax) - CurrentToDigits(a1Sum, IsumMax);
							outN2[j] += DigitsToAlgorithm(outputDigits, pSumMaxAlgorithm);
						}
						else {    // SRAM or digital eNVM
							int Dsum = 0;
							int DsumMax = 0;
							int a1Sum = 0;
							for (int k = 0; k < param->nHide; k++) {
								if ((da1[k] >> n) & 1) {    // if the nth bit of da1[k] is 1
									Dsum += (int)(arrayHO->ReadCell(j, k));
									a1Sum += pow(2, arrayHO->numCellPerSynapse) - 1;
								}
								DsumMax += pow(2, arrayHO->numCellPerSynapse) - 1;
							}
							if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayHO->cell[0][0])) {    // Digital eNVM
								sumArrayReadEnergy += static_cast<DigitalNVM*>(arrayHO->cell[0][0])->readEnergy * arrayHO->numCellPerSynapse * arrayHO->arrayRowSize;
							}
							else {
								sumArrayReadEnergy += static_cast<SRAM*>(arrayHO->cell[0][0])->readEnergy * arrayHO->numCellPerSynapse * arrayHO->arrayRowSize;
							}
							outN2[j] += (double)(2 * Dsum - a1Sum) / DsumMax * pSumMaxAlgorithm;
						}
					}
					a2[j] = sigmoid(outN2[j]);
				}
				arrayHO->readEnergy += sumArrayReadEnergy;

				numBatchReadSynapse = (int)ceil((double)param->nOutput / param->numColMuxed);
				// Don't parallelize this loop since there may be update of member variables inside NeuroSim functions
				for (int j = 0; j < param->nOutput; j += numBatchReadSynapse) {
					int numActiveRows = 0;  // Number of selected rows for NeuroSim
					for (int n = 0; n < param->numBitInput; n++) {
						for (int k = 0; k < param->nHide; k++) {
							if ((da1[k] >> n) & 1) {    // if the nth bit of da1[k] is 1
								numActiveRows++;
							}
						}
					}
					subArrayHO->activityRowRead = (double)numActiveRows / param->nHide / param->numBitInput;
					subArrayHO->readDynamicEnergy += NeuroSimSubArrayReadEnergy(subArrayHO);
					subArrayHO->readDynamicEnergy += NeuroSimNeuronReadEnergy(subArrayHO, adderHO, muxHO, muxDecoderHO, dffHO);
					subArrayHO->readLatency += NeuroSimSubArrayReadLatency(subArrayHO);
					subArrayHO->readLatency += NeuroSimNeuronReadLatency(subArrayHO, adderHO, muxHO, muxDecoderHO, dffHO);
				}
			}
			else {
#pragma omp parallel for
				for (int j = 0; j < param->nOutput; j++) {
					for (int k = 0; k < param->nHide; k++) {
						outN2[j] += 2 * a1[k] * weight2[j][k] - a1[k];
					}
					a2[j] = sigmoid(outN2[j]);
				}
			}

			// Backpropagation
			/* Second layer (hidder layer to the output layer) */
			for (int j = 0; j < param->nOutput; j++) {
				s2[j] = -2 * a2[j] * (1 - a2[j])*(Output[i][j] - a2[j]);
			}

			/* First layer (input layer to the hidden layer) */
			std::fill_n(s1, param->nHide, 0);
#pragma omp parallel for
			for (int j = 0; j < param->nHide; j++) {
				for (int k = 0; k < param->nOutput; k++) {
					s1[j] += a1[j] * (1 - a1[j]) * (2 * weight2[k][j] - 1) * s2[k];
				}
			}

			// Weight update
			/* Update weight of the first layer (input layer to the hidden layer) */
			if (param->useHardwareInTrainingWU) {
				double sumArrayWriteEnergy = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double sumNeuroSimWriteEnergy = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double sumWriteLatencyAnalogNVM = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double numWriteOperation = 0;	// Average number of write batches in the whole array. Use a temporary variable here since OpenMP does not support reduction on class member
				double writeVoltageLTP = static_cast<eNVM*>(arrayIH->cell[0][0])->writeVoltageLTP;
				double writeVoltageLTD = static_cast<eNVM*>(arrayIH->cell[0][0])->writeVoltageLTD;
				double writePulseWidthLTP = static_cast<eNVM*>(arrayIH->cell[0][0])->writePulseWidthLTP;
				double writePulseWidthLTD = static_cast<eNVM*>(arrayIH->cell[0][0])->writePulseWidthLTD;
				numBatchWriteSynapse = (int)ceil((double)arrayIH->arrayColSize / param->numWriteColMuxed);
#pragma omp parallel for reduction(+: sumArrayWriteEnergy, sumNeuroSimWriteEnergy, sumWriteLatencyAnalogNVM)
				for (int k = 0; k < param->nInput; k++) {
					int numWriteOperationPerRow = 0;	// Number of write batches in a row that have any weight change
					int numWriteCellPerOperation = 0;	// Average number of write cells per batch in a row (for digital eNVM)
					for (int j = 0; j < param->nHide; j += numBatchWriteSynapse) {
						/* Batch write */
						int start = j;
						int end = j + numBatchWriteSynapse - 1;
						if (end >= param->nHide) {
							end = param->nHide - 1;
						}
						double maxLatencyLTP = 0;	// Max latency for AnalogNVM's LTP or weight increase in this batch write
						double maxLatencyLTD = 0;	// Max latency for AnalogNVM's LTD or weight decrease in this batch write
						bool weightChangeBatch = false;	// Specify if there is any weight change in the entire write batch
						for (int jj = start; jj <= end; jj++) { // Selected cells
							deltaWeight1[jj][k] = -param->alpha1 * s1[jj] * Input[i][k];
							arrayIH->WriteCell(jj, k, deltaWeight1[jj][k], param->maxWeight, param->minWeight, true);
							//weight1[jj][k] += deltaWeight1[jj][k];
							weight1[jj][k] = arrayIH->ConductanceToWeight(jj, k, param->maxWeight, param->minWeight);//eltaWeight1[jj][k];
							//arrayIH->WriteCell(jj, k, deltaWeight1[jj][k], param->maxWeight, param->minWeight, true);
							//weight1[jj][k] = arrayIH->ConductanceToWeight(jj, k, param->maxWeight, param->minWeight);
							//std::cout << jj << "," << k << ":" << weight1[jj][k] << std::endl;
							if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[jj][k])) {	// Analog eNVM
								weightChangeBatch = weightChangeBatch || static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->numPulse;
								/* Get maxLatencyLTP and maxLatencyLTD */
								if (static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->writeLatencyLTP > maxLatencyLTP)
									maxLatencyLTP = static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->writeLatencyLTP;
								if (static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->writeLatencyLTD > maxLatencyLTD)
									maxLatencyLTD = static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->writeLatencyLTD;
							}
							else {	// SRAM and digital eNVM
								weightChangeBatch = weightChangeBatch || arrayIH->weightChange[jj][k];
							}
						}
						numWriteOperationPerRow += weightChangeBatch;
						for (int jj = start; jj <= end; jj++) { // Selected cells
							if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
								/* Set the max latency for all the selected cells in this batch */
								static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->writeLatencyLTP = maxLatencyLTP;
								static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->writeLatencyLTD = maxLatencyLTD;
								if (param->writeEnergyReport && weightChangeBatch) {
									if (static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->nonIdenticalPulse) {	// Non-identical write pulse scheme
										if (static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->numPulse > 0) {	// LTP
											static_cast<eNVM*>(arrayIH->cell[jj][k])->writeVoltageLTP = sqrt(static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->writeVoltageSquareSum / static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->numPulse);	// RMS value of LTP write voltage
											static_cast<eNVM*>(arrayIH->cell[jj][k])->writeVoltageLTD = static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->VinitLTD + 0.5 * static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->VstepLTD * static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->maxNumLevelLTD;	// Use average voltage of LTD write voltage
										}
										else if (static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->numPulse < 0) {	// LTD
											static_cast<eNVM*>(arrayIH->cell[jj][k])->writeVoltageLTP = static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->VinitLTP + 0.5 * static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->VstepLTP * static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->maxNumLevelLTP;    // Use average voltage of LTP write voltage
											static_cast<eNVM*>(arrayIH->cell[jj][k])->writeVoltageLTD = sqrt(static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->writeVoltageSquareSum / (-1 * static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->numPulse));    // RMS value of LTD write voltage
										}
										else {	// Half-selected during LTP and LTD phases
											static_cast<eNVM*>(arrayIH->cell[jj][k])->writeVoltageLTP = static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->VinitLTP + 0.5 * static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->VstepLTP * static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->maxNumLevelLTP;    // Use average voltage of LTP write voltage
											static_cast<eNVM*>(arrayIH->cell[jj][k])->writeVoltageLTD = static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->VinitLTD + 0.5 * static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->VstepLTD * static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->maxNumLevelLTD;    // Use average voltage of LTD write voltage
										}
									}
									static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->WriteEnergyCalculation(arrayIH->wireCapCol);
									sumArrayWriteEnergy += static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->writeEnergy;
								}
							}
							else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH->cell[0][0])) { // Digital eNVM
								if (param->writeEnergyReport && arrayIH->weightChange[jj][k]) {
									for (int n = 0; n < arrayIH->numCellPerSynapse; n++) {  // n=0 is LSB
										sumArrayWriteEnergy += static_cast<DigitalNVM*>(arrayIH->cell[(jj + 1) * arrayIH->numCellPerSynapse - (n + 1)][k])->writeEnergy;
										int bitPrev = static_cast<DigitalNVM*>(arrayIH->cell[(jj + 1) * arrayIH->numCellPerSynapse - (n + 1)][k])->bitPrev;
										int bit = static_cast<DigitalNVM*>(arrayIH->cell[(jj + 1) * arrayIH->numCellPerSynapse - (n + 1)][k])->bit;
										if (bit != bitPrev) {
											numWriteCellPerOperation += 1;
										}
									}
								}
							}
							else {    // SRAM
								if (param->writeEnergyReport && arrayIH->weightChange[jj][k]) {
									sumArrayWriteEnergy += static_cast<SRAM*>(arrayIH->cell[jj * arrayIH->numCellPerSynapse][k])->writeEnergy;
								}
							}
						}
						/* Latency for each batch write in Analog eNVM */
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {	// Analog eNVM
							sumWriteLatencyAnalogNVM += maxLatencyLTP + maxLatencyLTD;
						}
						/* Energy consumption on array caps for eNVM */
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
							if (param->writeEnergyReport && weightChangeBatch) {
								if (static_cast<AnalogNVM*>(arrayIH->cell[0][0])->nonIdenticalPulse) { // Non-identical write pulse scheme
									writeVoltageLTP = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->VinitLTP + 0.5 * static_cast<AnalogNVM*>(arrayIH->cell[0][0])->VstepLTP * static_cast<AnalogNVM*>(arrayIH->cell[0][0])->maxNumLevelLTP;    // Use average voltage of LTP write voltage
									writeVoltageLTD = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->VinitLTD + 0.5 * static_cast<AnalogNVM*>(arrayIH->cell[0][0])->VstepLTD * static_cast<AnalogNVM*>(arrayIH->cell[0][0])->maxNumLevelLTD;    // Use average voltage of LTD write voltage
								}
								if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
									// The energy on selected SLs is included in WriteCell()
									sumArrayWriteEnergy += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTP * writeVoltageLTP;   // Selected BL (LTP phases)
									sumArrayWriteEnergy += arrayIH->wireCapCol * writeVoltageLTP * writeVoltageLTP * (param->nHide - numBatchWriteSynapse);   // Unselected SLs (LTP phase)
									// No LTD part because all unselected rows and columns are V=0
								}
								else {
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTP * writeVoltageLTP;    // Selected WL (LTP phase)
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTP / 2 * writeVoltageLTP / 2 * (param->nInput - 1);  // Unselected WLs (LTP phase)
									sumArrayWriteEnergy += arrayIH->wireCapCol * writeVoltageLTP / 2 * writeVoltageLTP / 2 * (param->nHide - numBatchWriteSynapse);   // Unselected BLs (LTP phase)
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTD / 2 * writeVoltageLTD / 2 * (param->nInput - 1);    // Unselected WLs (LTD phase)
									sumArrayWriteEnergy += arrayIH->wireCapCol * writeVoltageLTD / 2 * writeVoltageLTD / 2 * (param->nHide - numBatchWriteSynapse); // Unselected BLs (LTD phase)
								}
							}
						}
						else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH->cell[0][0])) { // Digital eNVM
							if (param->writeEnergyReport && weightChangeBatch) {
								if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
									// The energy on selected columns is included in WriteCell()
									sumArrayWriteEnergy += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 for both SET and RESET phases)
								}
								else {    // Cross-point
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTP * writeVoltageLTP;   // Selected WL (SET phase)
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTP / 2 * writeVoltageLTP / 2 * (param->nInput - 1);    // Unselected WLs (SET phase)
									sumArrayWriteEnergy += arrayIH->wireCapCol * writeVoltageLTP / 2 * writeVoltageLTP / 2 * (param->nHide - numBatchWriteSynapse) * arrayIH->numCellPerSynapse;   // Unselected BLs (SET phase)
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTD / 2 * writeVoltageLTD / 2 * (param->nInput - 1);   // Unselected WLs (RESET phase)
									sumArrayWriteEnergy += arrayIH->wireCapCol * writeVoltageLTD / 2 * writeVoltageLTD / 2 * (param->nHide - numBatchWriteSynapse) * arrayIH->numCellPerSynapse;   // Unselected BLs (RESET phase)
								}
							}
						}
						/* Half-selected cells for eNVM */
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
							if (!static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess && param->writeEnergyReport) { // Cross-point
								for (int jj = 0; jj < param->nHide; jj++) { // Half-selected cells in the same row
									if (jj >= start && jj <= end) { continue; } // Skip the selected cells
									sumArrayWriteEnergy += (writeVoltageLTP / 2 * writeVoltageLTP / 2 * static_cast<eNVM*>(arrayIH->cell[jj][k])->conductanceAtHalfVwLTP * maxLatencyLTP + writeVoltageLTD / 2 * writeVoltageLTD / 2 * static_cast<eNVM*>(arrayIH->cell[jj][k])->conductanceAtHalfVwLTD * maxLatencyLTD);
								}
								for (int kk = 0; kk < param->nInput; kk++) {    // Half-selected cells in other rows
									// Note that here is a bit inaccurate if using OpenMP, because the weight on other rows (threads) are also being updated
									if (kk == k) { continue; } // Skip the selected row
									for (int jj = start; jj <= end; jj++) {
										sumArrayWriteEnergy += (writeVoltageLTP / 2 * writeVoltageLTP / 2 * static_cast<eNVM*>(arrayIH->cell[jj][kk])->conductanceAtHalfVwLTP * maxLatencyLTP + writeVoltageLTD / 2 * writeVoltageLTD / 2 * static_cast<eNVM*>(arrayIH->cell[jj][kk])->conductanceAtHalfVwLTD * maxLatencyLTD);
									}
								}
							}
						}
						else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH->cell[0][0])) { // Digital eNVM
							if (!static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess && param->writeEnergyReport && weightChangeBatch) { // Cross-point
								for (int jj = 0; jj < param->nHide; jj++) {    // Half-selected synapses in the same row
									if (jj >= start && jj <= end) { continue; } // Skip the selected synapses
									for (int n = 0; n < arrayIH->numCellPerSynapse; n++) {  // n=0 is LSB
										int colIndex = (jj + 1) * arrayIH->numCellPerSynapse - (n + 1);
										sumArrayWriteEnergy += writeVoltageLTP / 2 * writeVoltageLTP / 2 * static_cast<eNVM*>(arrayIH->cell[colIndex][k])->conductanceAtHalfVwLTP * maxLatencyLTP + writeVoltageLTD / 2 * writeVoltageLTD / 2 * static_cast<eNVM*>(arrayIH->cell[colIndex][k])->conductanceAtHalfVwLTD * maxLatencyLTD;
									}
								}
								for (int kk = 0; kk < param->nInput; kk++) {   // Half-selected synapses in other rows
									// Note that here is a bit inaccurate if using OpenMP, because the weight on other rows (threads) are also being updated
									if (kk == k) { continue; } // Skip the selected row
									for (int jj = start; jj <= end; jj++) {
										for (int n = 0; n < arrayIH->numCellPerSynapse; n++) {  // n=0 is LSB
											int colIndex = (jj + 1) * arrayIH->numCellPerSynapse - (n + 1);
											sumArrayWriteEnergy += writeVoltageLTP / 2 * writeVoltageLTP / 2 * static_cast<eNVM*>(arrayIH->cell[colIndex][kk])->conductanceAtHalfVwLTP * maxLatencyLTP + writeVoltageLTD / 2 * writeVoltageLTD / 2 * static_cast<eNVM*>(arrayIH->cell[colIndex][kk])->conductanceAtHalfVwLTD * maxLatencyLTD;
										}
									}
								}
							}
						}
					}
					/* Calculate the average number of write pulses on the selected row */
#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
					{
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
							int sumNumWritePulse = 0;
							for (int j = 0; j < param->nHide; j++) {
								sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayIH->cell[j][k])->numPulse);    // Note that LTD has negative pulse number
							}
							subArrayIH->numWritePulse = sumNumWritePulse / param->nHide;
							double writeVoltageSquareSumRow = 0;
							if (param->writeEnergyReport) {
								if (static_cast<AnalogNVM*>(arrayIH->cell[0][0])->nonIdenticalPulse) { // Non-identical write pulse scheme
									for (int j = 0; j < param->nHide; j++) {
										writeVoltageSquareSumRow += static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeVoltageSquareSum;
									}
									if (sumNumWritePulse > 0) {	// Prevent division by 0
										subArrayIH->cell.writeVoltage = sqrt(writeVoltageSquareSumRow / sumNumWritePulse);	// RMS value of write voltage in a row
									}
									else {
										subArrayIH->cell.writeVoltage = 0;
									}
								}
							}
						}
						numWriteCellPerOperation = (double)numWriteCellPerOperation / numWriteOperationPerRow;
						sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayIH, numWriteOperationPerRow, numWriteCellPerOperation);
					}
					numWriteOperation += numWriteOperationPerRow;
				}
				arrayIH->writeEnergy += sumArrayWriteEnergy;
				subArrayIH->writeDynamicEnergy += sumNeuroSimWriteEnergy;
				numWriteOperation = numWriteOperation / param->nInput;
				subArrayIH->writeLatency += NeuroSimSubArrayWriteLatency(subArrayIH, numWriteOperation, sumWriteLatencyAnalogNVM);
			}
			else {
#pragma omp parallel for
				for (int j = 0; j < param->nHide; j++) {
					for (int k = 0; k < param->nInput; k++) {
						deltaWeight1[j][k] = -param->alpha1 * s1[j] * Input[i][k];
						weight1[j][k] = weight1[j][k] + deltaWeight1[j][k];
						if (weight1[j][k] > param->maxWeight) {
							deltaWeight1[j][k] -= weight1[j][k] - param->maxWeight;
							weight1[j][k] = param->maxWeight;
						}
						else if (weight1[j][k] < param->minWeight) {
							deltaWeight1[j][k] += param->minWeight - weight1[j][k];
							weight1[j][k] = param->minWeight;
						}
						if (param->useHardwareInTrainingFF) {
							arrayIH->WriteCell(j, k, deltaWeight1[j][k], param->maxWeight, param->minWeight, false);
						}
					}
				}
			}

			/* Update weight of the second layer (hidden layer to the output layer) */
			if (param->useHardwareInTrainingWU) {
				double sumArrayWriteEnergy = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double sumNeuroSimWriteEnergy = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double sumWriteLatencyAnalogNVM = 0;	// Use a temporary variable here since OpenMP does not support reduction on class member
				double numWriteOperation = 0;	// Average number of write batches in the whole array. Use a temporary variable here since OpenMP does not support reduction on class member
				double writeVoltageLTP = static_cast<eNVM*>(arrayHO->cell[0][0])->writeVoltageLTP;
				double writeVoltageLTD = static_cast<eNVM*>(arrayHO->cell[0][0])->writeVoltageLTD;
				double writePulseWidthLTP = static_cast<eNVM*>(arrayHO->cell[0][0])->writePulseWidthLTP;
				double writePulseWidthLTD = static_cast<eNVM*>(arrayHO->cell[0][0])->writePulseWidthLTD;
				numBatchWriteSynapse = (int)ceil((double)arrayHO->arrayColSize / param->numWriteColMuxed);
				#pragma omp parallel for reduction(+: sumArrayWriteEnergy, sumNeuroSimWriteEnergy, sumWriteLatencyAnalogNVM)
				for (int k = 0; k < param->nHide; k++) {
					int numWriteOperationPerRow = 0;    // Number of write batches in a row that have any weight change
					int numWriteCellPerOperation = 0;   // Average number of write cells per batch in a row (for digital eNVM)
					for (int j = 0; j < param->nOutput; j += numBatchWriteSynapse) {
						/* Batch write */
						int start = j;
						int end = j + numBatchWriteSynapse - 1;
						if (end >= param->nOutput) {
							end = param->nOutput - 1;
						}
						double maxLatencyLTP = 0;   // Max latency for AnalogNVM's LTP or weight increase in this batch write
						double maxLatencyLTD = 0;   // Max latency for AnalogNVM's LTD or weight decrease in this batch write
						bool weightChangeBatch = false; // Specify if there is any weight change in the entire write batch
						for (int jj = start; jj <= end; jj++) { // Selected cells
							deltaWeight2[jj][k] = -param->alpha2 * s2[jj] * a1[k];
							arrayHO->WriteCell(jj, k, deltaWeight2[jj][k], param->maxWeight, param->minWeight, true);
							//double conductanceGp = static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->conductanceGp;
							//std::cout << conductanceGp << std::endl;
							weight2[jj][k] = arrayHO->ConductanceToWeight(jj, k, param->maxWeight, param->minWeight);//+deltaWeight2[jj][k];
							//weight2[jj][k] += deltaWeight2[jj][k];
							if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[jj][k])) { // Analog eNVM
								weightChangeBatch = weightChangeBatch || static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->numPulse;
								/* Get maxLatencyLTP and maxLatencyLTD */
								if (static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->writeLatencyLTP > maxLatencyLTP)
									maxLatencyLTP = static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->writeLatencyLTP;
								if (static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->writeLatencyLTD > maxLatencyLTD)
									maxLatencyLTD = static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->writeLatencyLTD;
							}
							else {    // SRAM and digital eNVM
								weightChangeBatch = weightChangeBatch || arrayHO->weightChange[jj][k];
							}
						}
						numWriteOperationPerRow += weightChangeBatch;
						for (int jj = start; jj <= end; jj++) { // Selected cells
							if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
								/* Set the max latency for all the cells in this batch */
								static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->writeLatencyLTP = maxLatencyLTP;
								static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->writeLatencyLTD = maxLatencyLTD;
								if (param->writeEnergyReport && weightChangeBatch) {
									if (static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->nonIdenticalPulse) { // Non-identical write pulse scheme
										if (static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->numPulse > 0) {  // LTP
											static_cast<eNVM*>(arrayHO->cell[jj][k])->writeVoltageLTP = sqrt(static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->writeVoltageSquareSum / static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->numPulse);   // RMS value of LTP write voltage
											static_cast<eNVM*>(arrayHO->cell[jj][k])->writeVoltageLTD = static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->VinitLTD + 0.5 * static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->VstepLTD * static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->maxNumLevelLTD;    // Use average voltage of LTD write voltage
										}
										else if (static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->numPulse < 0) {    // LTD
											static_cast<eNVM*>(arrayHO->cell[jj][k])->writeVoltageLTP = static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->VinitLTP + 0.5 * static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->VstepLTP * static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->maxNumLevelLTP;    // Use average voltage of LTP write voltage
											static_cast<eNVM*>(arrayHO->cell[jj][k])->writeVoltageLTD = sqrt(static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->writeVoltageSquareSum / (-1 * static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->numPulse));    // RMS value of LTD write voltage
										}
										else {	// Half-selected during LTP and LTD phases
											static_cast<eNVM*>(arrayHO->cell[jj][k])->writeVoltageLTP = static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->VinitLTP + 0.5 * static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->VstepLTP * static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->maxNumLevelLTP;    // Use average voltage of LTP write voltage
											static_cast<eNVM*>(arrayHO->cell[jj][k])->writeVoltageLTD = static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->VinitLTD + 0.5 * static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->VstepLTD * static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->maxNumLevelLTD;    // Use average voltage of LTD write voltage
										}
									}
									static_cast<AnalogNVM*>(arrayHO->cell[jj][k])->WriteEnergyCalculation(arrayHO->wireCapCol);
									sumArrayWriteEnergy += static_cast<eNVM*>(arrayHO->cell[jj][k])->writeEnergy;
								}
							}
							else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayHO->cell[0][0])) { // Digital eNVM
								if (param->writeEnergyReport && arrayHO->weightChange[jj][k]) {
									for (int n = 0; n < arrayHO->numCellPerSynapse; n++) {  // n=0 is LSB
										sumArrayWriteEnergy += static_cast<DigitalNVM*>(arrayHO->cell[(jj + 1) * arrayHO->numCellPerSynapse - (n + 1)][k])->writeEnergy;
										int bitPrev = static_cast<DigitalNVM*>(arrayHO->cell[(jj + 1) * arrayHO->numCellPerSynapse - (n + 1)][k])->bitPrev;
										int bit = static_cast<DigitalNVM*>(arrayHO->cell[(jj + 1) * arrayHO->numCellPerSynapse - (n + 1)][k])->bit;
										if (bit != bitPrev) {
											numWriteCellPerOperation += 1;
										}
									}
								}
							}
							else {    // SRAM
								if (param->writeEnergyReport && arrayHO->weightChange[jj][k]) {
									sumArrayWriteEnergy += static_cast<SRAM*>(arrayHO->cell[jj * arrayHO->numCellPerSynapse][k])->writeEnergy;
								}
							}
						}
						/* Latency for each batch write in Analog eNVM */
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
							sumWriteLatencyAnalogNVM += maxLatencyLTP + maxLatencyLTD;
						}
						/* Energy consumption on array caps for eNVM */
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
							if (param->writeEnergyReport && weightChangeBatch) {
								if (static_cast<AnalogNVM*>(arrayHO->cell[0][0])->nonIdenticalPulse) { // Non-identical write pulse scheme
									writeVoltageLTP = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->VinitLTP + 0.5 * static_cast<AnalogNVM*>(arrayHO->cell[0][0])->VstepLTP * static_cast<AnalogNVM*>(arrayHO->cell[0][0])->maxNumLevelLTP;    // Use average voltage of LTP write voltage
									writeVoltageLTD = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->VinitLTD + 0.5 * static_cast<AnalogNVM*>(arrayHO->cell[0][0])->VstepLTD * static_cast<AnalogNVM*>(arrayHO->cell[0][0])->maxNumLevelLTD;    // Use average voltage of LTD write voltage
								}
								if (static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) {  // 1T1R
									// The energy on selected SLs is included in WriteCell()
									sumArrayWriteEnergy += arrayHO->wireGateCapRow * techHO.vdd * techHO.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTP * writeVoltageLTP;   // Selected BL (LTP phases)
									sumArrayWriteEnergy += arrayHO->wireCapCol * writeVoltageLTP * writeVoltageLTP * (param->nOutput - numBatchWriteSynapse);   // Unselected SLs (LTP phase)
									// No LTD part because all unselected rows and columns are V=0
								}
								else {
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTP * writeVoltageLTP;   // Selected WL (LTP phase)
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTP / 2 * writeVoltageLTP / 2 * (param->nHide - 1);    // Unselected WLs (LTP phase)
									sumArrayWriteEnergy += arrayHO->wireCapCol * writeVoltageLTP / 2 * writeVoltageLTP / 2 * (param->nOutput - numBatchWriteSynapse); // Unselected BLs (LTP phase)
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTD / 2 * writeVoltageLTD / 2 * (param->nHide - 1);    // Unselected WLs (LTD phase)
									sumArrayWriteEnergy += arrayHO->wireCapCol * writeVoltageLTD / 2 * writeVoltageLTD / 2 * (param->nOutput - numBatchWriteSynapse); // Unselected BLs (LTD phase)
								}
							}
						}
						else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayHO->cell[0][0])) { // Digital eNVM
							if (param->writeEnergyReport && weightChangeBatch) {
								if (static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) {  // 1T1R
									// The energy on selected columns is included in WriteCell()
									sumArrayWriteEnergy += arrayHO->wireGateCapRow * techHO.vdd * techHO.vdd * 2;   // Selected WL (*2 for both SET and RESET phases)
								}
								else {    // Cross-point
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTP * writeVoltageLTP;   // Selected WL (SET phase)
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTP / 2 * writeVoltageLTP / 2 * (param->nInput - 1);    // Unselected WLs (SET phase)
									sumArrayWriteEnergy += arrayHO->wireCapCol * writeVoltageLTP / 2 * writeVoltageLTP / 2 * (param->nHide - numBatchWriteSynapse) * arrayHO->numCellPerSynapse;  // Unselected BLs (SET phase)
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTD / 2 * writeVoltageLTD / 2 * (param->nInput - 1);  // Unselected WLs (RESET phase)
									sumArrayWriteEnergy += arrayHO->wireCapCol * writeVoltageLTD / 2 * writeVoltageLTD / 2 * (param->nHide - numBatchWriteSynapse) * arrayHO->numCellPerSynapse;  // Unselected BLs (RESET phase)
								}
							}
						}
						/* Half-selected cells for eNVM */
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
							if (!static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess && param->writeEnergyReport) { // Cross-point
								for (int jj = 0; jj < param->nOutput; jj++) {    // Half-selected cells in the same row
									if (jj >= start && jj <= end) { continue; } // Skip the selected cells
									sumArrayWriteEnergy += (writeVoltageLTP / 2 * writeVoltageLTP / 2 * static_cast<eNVM*>(arrayHO->cell[jj][k])->conductanceAtHalfVwLTP * maxLatencyLTP + writeVoltageLTD / 2 * writeVoltageLTD / 2 * static_cast<eNVM*>(arrayHO->cell[jj][k])->conductanceAtHalfVwLTD * maxLatencyLTD);
								}
								for (int kk = 0; kk < param->nHide; kk++) { // Half-selected cells in other rows
									// Note that here is a bit inaccurate if using OpenMP, because the weight on other rows (threads) are also being updated
									if (kk == k) { continue; }  // Skip the selected row
									for (int jj = start; jj <= end; jj++) {
										sumArrayWriteEnergy += (writeVoltageLTP / 2 * writeVoltageLTP / 2 * static_cast<eNVM*>(arrayHO->cell[jj][kk])->conductanceAtHalfVwLTP * maxLatencyLTP + writeVoltageLTD / 2 * writeVoltageLTD / 2 * static_cast<eNVM*>(arrayHO->cell[jj][kk])->conductanceAtHalfVwLTD * maxLatencyLTD);
									}
								}
							}
						}
						else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayHO->cell[0][0])) { // Digital eNVM
							if (!static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess && param->writeEnergyReport && weightChangeBatch) { // Cross-point
								for (int jj = 0; jj < param->nOutput; jj++) {    // Half-selected synapses in the same row
									if (jj >= start && jj <= end) { continue; } // Skip the selected synapses
									for (int n = 0; n < arrayHO->numCellPerSynapse; n++) {  // n=0 is LSB
										int colIndex = (jj + 1) * arrayHO->numCellPerSynapse - (n + 1);
										sumArrayWriteEnergy += writeVoltageLTP / 2 * writeVoltageLTP / 2 * static_cast<eNVM*>(arrayHO->cell[colIndex][k])->conductanceAtHalfVwLTP * maxLatencyLTP + writeVoltageLTD / 2 * writeVoltageLTD / 2 * static_cast<eNVM*>(arrayHO->cell[colIndex][k])->conductanceAtHalfVwLTD * maxLatencyLTD;
									}
								}
								for (int kk = 0; kk < param->nHide; kk++) {    // Half-selected synapses in other rows
									// Note that here is a bit inaccurate if using OpenMP, because the weight on other rows (threads) are also being updated
									if (kk == k) { continue; }  // Skip the selected row
									for (int jj = start; jj <= end; jj++) {
										for (int n = 0; n < arrayHO->numCellPerSynapse; n++) {  // n=0 is LSB
											int colIndex = (jj + 1) * arrayHO->numCellPerSynapse - (n + 1);
											sumArrayWriteEnergy += writeVoltageLTP / 2 * writeVoltageLTP / 2 * static_cast<eNVM*>(arrayHO->cell[colIndex][kk])->conductanceAtHalfVwLTP * maxLatencyLTP + writeVoltageLTD / 2 * writeVoltageLTD / 2 * static_cast<eNVM*>(arrayHO->cell[colIndex][kk])->conductanceAtHalfVwLTD * maxLatencyLTD;
										}
									}
								}
							}
						}
					}
					/* Calculate the average number of write pulses on the selected row */
#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
					{
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
							int sumNumWritePulse = 0;
							for (int j = 0; j < param->nOutput; j++) {
								sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayHO->cell[j][k])->numPulse);    // Note that LTD has negative pulse number
							}
							subArrayHO->numWritePulse = sumNumWritePulse / param->nOutput;
							double writeVoltageSquareSumRow = 0;
							if (param->writeEnergyReport) {
								if (static_cast<AnalogNVM*>(arrayHO->cell[0][0])->nonIdenticalPulse) { // Non-identical write pulse scheme
									for (int j = 0; j < param->nOutput; j++) {
										writeVoltageSquareSumRow += static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeVoltageSquareSum;
									}
									if (sumNumWritePulse > 0) {	// Prevent division by 0
										subArrayHO->cell.writeVoltage = sqrt(writeVoltageSquareSumRow / sumNumWritePulse);  // RMS value of write voltage in a row
									}
									else {
										subArrayHO->cell.writeVoltage = 0;
									}
								}
							}
						}
						numWriteCellPerOperation = (double)numWriteCellPerOperation / numWriteOperationPerRow;
						sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayHO, numWriteOperationPerRow, numWriteCellPerOperation);
					}
					numWriteOperation += numWriteOperationPerRow;
				}

				arrayHO->writeEnergy += sumArrayWriteEnergy;
				subArrayHO->writeDynamicEnergy += sumNeuroSimWriteEnergy;
				numWriteOperation = numWriteOperation / param->nHide;
				subArrayHO->writeLatency += NeuroSimSubArrayWriteLatency(subArrayHO, numWriteOperation, sumWriteLatencyAnalogNVM);
			}
			else {
#pragma omp parallel for
				for (int j = 0; j < param->nOutput; j++) {
					for (int k = 0; k < param->nHide; k++) {
						deltaWeight2[j][k] = -param->alpha2 * s2[j] * a1[k];
						weight2[j][k] = weight2[j][k] + deltaWeight2[j][k];
						if (weight2[j][k] > param->maxWeight) {
							deltaWeight2[j][k] -= weight2[j][k] - param->maxWeight;
							weight2[j][k] = param->maxWeight;
						}
						else if (weight2[j][k] < param->minWeight) {
							deltaWeight2[j][k] += param->minWeight - weight2[j][k];
							weight2[j][k] = param->minWeight;
						}
						if (param->useHardwareInTrainingFF) {
							arrayHO->WriteCell(j, k, deltaWeight2[j][k], param->maxWeight, param->minWeight, false);
						}
					}
				}
			}
			/*======================================PCM Operation===============================*/

			if (param->useHardwareInTraining) {
				if (static_cast<AnalogNVM*>(arrayIH->cell[0][0])->PCMON) {
					if (((batchSize + 1) % param->numImageperRESET == 0)) { //occational RESET numImageperRESET=100
						if (param->RandomRefresh) { //Refresh random cell			
							// Line 별 Refresh를 진행
							/*Read All first Layer*/
							if(param->mode == 0){ // Line mode
								std::mt19937 Randgen;
								Randgen.seed(std::time(0));
								double RandNum = 0;
								int Ref[param->nHide];
								int count1 = 0;
								int num1, num2, temp = 0;
								std::uniform_real_distribution<double> dist(0, 1);
								std::uniform_int_distribution<int> distIH(0, param->nHide - 1);
								std::fill_n(Ref, param->nHide, 0);
								double maxConductance = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->maxConductance;
								double ResetThr = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->ThrConductance;
								double sumArrayReadEnergy = 0; // Read Energy를 더할 임시 변수
								double readVoltage = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->readVoltage;
								double readPulseWidth = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->readPulseWidth;
								#pragma omp parallel for reduction(+: sumArrayReadEnergy)
								for (int j = 0; j < param->nHide; j++) {
									if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) { //Analog PCM
										if (static_cast<AnalogNVM*>(arrayIH->cell[0][0])->cmosAccess) { //1T1R
											sumArrayReadEnergy += arrayIH->wireCapRow*techIH.vdd*techIH.vdd*param->nInput;// All WLs open
										}
									}
									Ref[j] = j;
								}
								for (int a = 0; a < param->NumRefHiddenLayer * 30; a++) { //Shuffle 과정
									num1 = distIH(Randgen);
									num2 = distIH(Randgen);
									temp = Ref[num1];
									Ref[num1] = Ref[num2];
									Ref[num2] = temp;
								}
								for (int k = 0; k < param->nInput; k++) {
									for (int j = 0; j < param->nHide; j++) {
										//std::cout << RandNum << std::endl;
										for (int a = 0; a < param->NumRefHiddenLayer; a++) {
											if (j == Ref[a]) {
												static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM = true;
												count1 += 1;
											}
										}
									}

								}
								//std::cout << count1 << std::endl;
								//std::cout << count1 << std :: endl;
								/*Read Second Layer*/
								int Ref2[param->nOutput];
								int count2 = 0;
								num1, num2, temp = 0;
								//std::uniform_real_distribution<double> dist(0, 1);
								std::uniform_int_distribution<int> distIH2(0, param->nOutput - 1);
								std::fill_n(Ref2, param->nOutput, 0);
								maxConductance = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->maxConductance;
								ResetThr = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->ThrConductance;
								sumArrayReadEnergy = 0; // Read Energy를 더할 임시 변수
								readVoltage = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->readVoltage;
								readPulseWidth = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->readPulseWidth;
#pragma omp parallel for reduction(+: sumArrayReadEnergy)
								for (int j = 0; j < param->nOutput; j++) {
									if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) { //Analog PCM
										if (static_cast<AnalogNVM*>(arrayHO->cell[0][0])->cmosAccess) { //1T1R
											sumArrayReadEnergy += arrayHO->wireCapRow*techIH.vdd*techIH.vdd*param->nHide;// All WLs open
										}
									}
									Ref2[j] = j;
								}
								for (int a = 0; a < param->NumRefOutputLayer * 30; a++) { //Shuffle 과정
									num1 = distIH2(Randgen);
									num2 = distIH2(Randgen);
									temp = Ref2[num1];
									Ref2[num1] = Ref2[num2];
									Ref2[num2] = temp;
								}
								for (int k = 0; k < param->nHide; k++) {
									for (int j = 0; j < param->nOutput; j++) {
										//std::cout << RandNum << std::endl;
										for (int a = 0; a < param->NumRefOutputLayer; a++) {
											if (j == Ref2[a]) {
												static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM = true;
												//count2+=1;
											}
										}



									}

								}
								//std::cout << count2 << std::endl;
								arrayHO->readEnergy += sumArrayReadEnergy;
								// Don't parallelize this loop since there may be update of member variables inside NeuroSim functions
								subArrayHO->activityRowRead = 1;
								subArrayHO->readDynamicEnergy += NeuroSimSubArrayReadEnergy(subArrayHO);
								subArrayHO->readDynamicEnergy += NeuroSimNeuronReadEnergy(subArrayHO, adderIH, muxIH, muxDecoderIH, dffIH);
								subArrayHO->readLatency += NeuroSimSubArrayReadLatency(subArrayHO);
								subArrayHO->readLatency += NeuroSimNeuronReadLatency(subArrayHO, adderIH, muxIH, muxDecoderIH, dffIH);
							}
							else if (param->mode == 1) { // Sporadic
								std::mt19937 Randgen;
								Randgen.seed(std::time(0));
								double RandNum = 0;
								int count1 = 0;
								std::uniform_real_distribution<double> dist(0, 1);
								double maxConductance = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->maxConductance;
								double ResetThr = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->ThrConductance;
								double sumArrayReadEnergy = 0; // Read Energy를 더할 임시 변수
								double readVoltage = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->readVoltage;
								double readPulseWidth = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->readPulseWidth;
									#pragma omp parallel for reduction(+: sumArrayReadEnergy)
								for (int j = 0; j < param->nHide; j++) {
									if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) { //Analog PCM
										if (static_cast<AnalogNVM*>(arrayIH->cell[0][0])->cmosAccess) { //1T1R
											sumArrayReadEnergy += arrayIH->wireCapRow*techIH.vdd*techIH.vdd*param->nInput;// All WLs open
										}
									}
									for (int n = 0; n < param->numBitInput; n++) {
										double pSumMaxAlgorithm = pow(2, n) / (param->numInputLevel - 1)*arrayIH->arrayRowSize; // numInputLevel= 2 (black or white)
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {
											double Isum = 0; // weight sum current
											double IsumMax = 0; //Max weight sum current
											double inputSum = 0;  // weight sum current of input vector
											for (int k = 0; k < param->nInput; k++) {
												RandNum = dist(Randgen);
												Isum += arrayIH->ReadCell(j, k);
												if (RandNum > param->ActDeviceIH) {
													static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM = true;
												}
												else {
													static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM = false;
												}
											}
											sumArrayReadEnergy += Isum * readVoltage*readPulseWidth; //Read energy를 다 더해줌
										}

									}
								}
								arrayIH->readEnergy += sumArrayReadEnergy;
								// Don't parallelize this loop since there may be update of member variables inside NeuroSim functions
								subArrayIH->activityRowRead = 1;
								subArrayIH->readDynamicEnergy += NeuroSimSubArrayReadEnergy(subArrayIH);
								subArrayIH->readDynamicEnergy += NeuroSimNeuronReadEnergy(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
								subArrayIH->readLatency += NeuroSimSubArrayReadLatency(subArrayIH);
								subArrayIH->readLatency += NeuroSimNeuronReadLatency(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);

								/*Read All Second Layer*/
								maxConductance = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->maxConductance;
								ResetThr = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->ThrConductance;
								sumArrayReadEnergy = 0; // Read Energy를 더할 임시 변수
								readVoltage = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->readVoltage;
								readPulseWidth = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->readPulseWidth;
#pragma omp parallel for reduction(+: sumArrayReadEnergy)
								for (int j = 0; j < param->nOutput; j++) {
									if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) { //Analog PCM
										if (static_cast<AnalogNVM*>(arrayHO->cell[0][0])->cmosAccess) { //1T1R
											sumArrayReadEnergy += arrayHO->wireCapRow*techIH.vdd*techIH.vdd*param->nInput;// All WLs open
										}
									}
									for (int n = 0; n < param->numBitInput; n++) {
										double pSumMaxAlgorithm = pow(2, n) / (param->numInputLevel - 1)*(arrayHO->arrayRowSize); // numInputLevel= 2 (black or white)
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {
											RandNum = dist(Randgen);
											double Isum = 0; // weight sum current
											double IsumMax = 0; //Max weight sum current
											double inputSum = 0;  // weight sum current of input vector
											for (int k = 0; k < param->nHide; k++) {
												//std::cout << RandNum
												Isum += arrayHO->ReadCell(j, k);
												if (RandNum > param->ActDeviceHO) {
													static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM = true;
												}
												else {
													static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM = false;
												}
												/*	else
													{
														static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM = false;

													}*/
											}
											sumArrayReadEnergy += Isum * readVoltage*readPulseWidth;
										}

									}
								}
								arrayHO->readEnergy += sumArrayReadEnergy;
								// Don't parallelize this loop since there may be update of member variables inside NeuroSim functions
								subArrayHO->activityRowRead = 1;
								subArrayHO->readDynamicEnergy += NeuroSimSubArrayReadEnergy(subArrayIH);
								subArrayHO->readDynamicEnergy += NeuroSimNeuronReadEnergy(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
								subArrayHO->readLatency += NeuroSimSubArrayReadLatency(subArrayIH);
								subArrayHO->readLatency += NeuroSimNeuronReadLatency(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
							}
							else if (param->mode == 2) { // Sequential
							int count1 = 0;
							int batchNum = (batchSize + 1) / param->numImageperRESET;
							double maxConductance = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->maxConductance;
							double ResetThr = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->ThrConductance;
							double sumArrayReadEnergy = 0; // Read Energy를 더할 임시 변수
							double readVoltage = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->readVoltage;
							double readPulseWidth = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->readPulseWidth;
							int RefStart;
							int RefEnd;
						
							int RefLayer[param->NumRefHiddenLayer];
							std::fill_n(RefLayer, param->NumRefHiddenLayer, 0);
							RefStart = (batchNum*param->NumRefHiddenLayer);
							//std::cout << RefStart << std::endl;
							RefEnd = ((batchSize + 1)*param->NumRefHiddenLayer);
							for (int i = 0; i < param->NumRefHiddenLayer; i++) {
								RefLayer[i] = (RefStart + i)%param->nHide;
								//std::cout << RefLayer[i] << std::endl;
							}
								#pragma omp parallel for reduction(+: sumArrayReadEnergy)
							for (int j = 0; j < param->nHide; j++) {
								if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) { //Analog PCM
									if (static_cast<AnalogNVM*>(arrayIH->cell[0][0])->cmosAccess) { //1T1R
										sumArrayReadEnergy += arrayIH->wireCapRow*techIH.vdd*techIH.vdd*param->nInput;// All WLs open
									}
								}
							}
							for (int k = 0; k < param->nInput; k++) {
								for (int j = 0; j < param->nHide; j++) {
									//std::cout << RandNum << std::endl;
									for (int a = 0; a < param->NumRefHiddenLayer; a++) {
										if (j == RefLayer[a]) {
										
											static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM = true;
											//count1 += 1;
											//std::cout << j<<std::endl;
										}
									}

								}

							}
							//std::cout << count1 << std::endl;
							//std::cout << count1 << std :: endl;
							/*Read Second Layer*/
							RefStart;
							RefEnd;
							int RefLayer2[param->NumRefOutputLayer];
						
							std::fill_n(RefLayer2, param->NumRefOutputLayer, 0);
							RefStart = (batchNum*param->NumRefOutputLayer);
							RefEnd = ((batchSize + 1)*param->NumRefOutputLayer);
							for (int i = 0; i < param->NumRefOutputLayer; i++) {
								RefLayer2[i] = (RefStart + i) % param->nOutput;
							}
							#pragma omp parallel for reduction(+: sumArrayReadEnergy)
							for (int j = 0; j < param->nHide; j++) {
								if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) { //Analog PCM
									if (static_cast<AnalogNVM*>(arrayIH->cell[0][0])->cmosAccess) { //1T1R
										sumArrayReadEnergy += arrayIH->wireCapRow*techIH.vdd*techIH.vdd*param->nInput;// All WLs open
									}
								}
							}
							for (int k = 0; k < param->nHide; k++) {
								for (int j = 0; j < param->nOutput; j++) {
									//std::cout << RandNum << std::endl;

									for (int a = 0; a < param->NumRefOutputLayer; a++) {
										if (j == RefLayer2[a]) {
											
											static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM = true;
										}
									}

								}
							}
							
							arrayHO->readEnergy += sumArrayReadEnergy;
							// Don't parallelize this loop since there may be update of member variables inside NeuroSim functions
							subArrayHO->activityRowRead = 1;
							subArrayHO->readDynamicEnergy += NeuroSimSubArrayReadEnergy(subArrayHO);
							subArrayHO->readDynamicEnergy += NeuroSimNeuronReadEnergy(subArrayHO, adderIH, muxIH, muxDecoderIH, dffIH);
							subArrayHO->readLatency += NeuroSimSubArrayReadLatency(subArrayHO);
							subArrayHO->readLatency += NeuroSimNeuronReadLatency(subArrayHO, adderIH, muxIH, muxDecoderIH, dffIH);
							}

							/*ERASE Opeartion*/
							/*==================Erase First Layer===================*/
							double sumArrayWriteEnergy = 0;
							double sumNeuroSimWriteEnergy = 0;
							double sumWriteLatencyAnalogPCM = 0;
							double numWriteOperation = 0;
							double RESETVoltage = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->RESETVoltage;
							double RESETPulseWidth = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->RESETPulseWidth;
							int count4 = 0;
							#pragma omp parallel for reduction(+:sumArrayWriteEnergy,sumNeuroSimWriteEnergy,sumWriteLatencyAnalogPCM)
							for (int k = 0; k < param->nInput; k++) {
								int numWriteOperationPerRow = 0;
								int numWriteCellPerOperation = 0;
								double maxLatencyLTP = 0;
								for (int j = 0; j < param->nHide; j++) {
									if (static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM) { //SET Saturation 된 경우 RESET operation을 진행
										arrayIH->EraseCell(j, k, param->maxWeight, param->minWeight);
										//count4 += 1;
										numWriteCellPerOperation += 1;
										if (static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeLatencyLTP > maxLatencyLTP) {
											maxLatencyLTP = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeLatencyLTP;
										}
										if (param->writeEnergyReport) {
											static_cast<AnalogNVM*>(arrayIH->cell[j][k])->EraseEnergyCalculation(arrayIH->wireCapCol);
										}
										sumArrayWriteEnergy += static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeEnergy;
										/* Latency for each batch write in Analog eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {	// Analog eNVM
											sumWriteLatencyAnalogPCM += maxLatencyLTP;
										}
										/* Energy consumption on array caps for eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
											if (param->writeEnergyReport) {
												if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
													// The energy on selected SLs is included in WriteCell()
													sumArrayWriteEnergy += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage * RESETVoltage;   // Selected BL (LTP phases)
													sumArrayWriteEnergy += arrayIH->wireCapCol * RESETVoltage * RESETVoltage * (param->nHide - numBatchWriteSynapse);   // Unselected SLs (LTP phase)
													// No LTD part because all unselected rows and columns are V=0
												}
												else {
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage *RESETVoltage;    // Selected WL (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);  // Unselected WLs (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapCol *RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse);   // Unselected BLs (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);    // Unselected WLs (LTD phase)
													sumArrayWriteEnergy += arrayIH->wireCapCol * RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse); // Unselected BLs (LTD phase)
												}
											}
										}
									}
									/*Half selected Cell*/
									else { //Saturation이 안된경우는 그냥 남아있음.
										count4 += 1;
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
											if (!static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess && param->writeEnergyReport) { // Cross-point
												sumArrayWriteEnergy += (RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP + RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP);
											}
										}
									}
								}
							}
							//std::cout << count4 << std::endl;
							/* Calculate the average number of write pulses on the selected row */
				//		#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables 문제가능성 큼
				//	{
				//		if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
				//			int sumNumWritePulse = 0;
				//			for (int j = 0; j < param->nHide; j++) {
				//				sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayIH->cell[j][k])->numPulse);    // Note that LTD has negative pulse number/										}
				//				subArrayIH->numWritePulse = sumNumWritePulse / param->nHide;
				//				double writeVoltageSquareSumRow = 0;
				//			}
				//			numWriteCellPerOperation = (double)numWriteCellPerOperation / numWriteOperationPerRow;
				//			sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayIH, numWriteOperationPerRow, numWriteCellPerOperation);
				//		}
				//		numWriteOperation += numWriteOperationPerRow;
				//	}
				//}
							arrayIH->writeEnergy += sumArrayWriteEnergy;
							subArrayIH->writeDynamicEnergy += sumNeuroSimWriteEnergy;
							numWriteOperation = numWriteOperation / param->nInput;
							subArrayIH->writeLatency += NeuroSimSubArrayWriteLatency(subArrayIH, numWriteOperation, sumWriteLatencyAnalogPCM);

							/*==================Erase Second Layer===================*/
							sumArrayWriteEnergy = 0;
							sumNeuroSimWriteEnergy = 0;
							sumWriteLatencyAnalogPCM = 0;
							numWriteOperation = 0;
							RESETVoltage = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->RESETVoltage;
							RESETPulseWidth = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->RESETPulseWidth;
							int count3 = 0;
							/*double Gp = 0;
							double Gn = 0;*/
							#pragma omp parallel for reduction(+:sumArrayWriteEnergy,sumNeuroSimWriteEnergy,sumWriteLatencyAnalogPCM)
						
							for (int k = 0; k < param->nHide; k++) {
								int numWriteOperationPerRow = 0;
								int numWriteCellPerOperation = 0;
								double maxLatencyLTP = 0;

								for (int j = 0; j < param->nOutput; j++) {

									if (static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM) { //SET Saturation 된 경우 RESET operation을 진행
										count3 += 1;
										arrayHO->EraseCell(j, k, param->maxWeight, param->minWeight);
										/*double Gp = static_cast<AnalogNVM*>(arrayHO->cell[j][k])->conductanceGp;
										double Gn = static_cast<AnalogNVM*>(arrayHO->cell[j][k])->conductanceGn;
										std::cout << "GP: " << Gp << "GN: " << Gn << std::endl;*/
										numWriteCellPerOperation += 1;
										if (static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeLatencyLTP > maxLatencyLTP) {
											maxLatencyLTP = static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeLatencyLTP;
										}
										if (param->writeEnergyReport) {
											//static_cast<AnalogNVM*>(arrayHO->cell[j][k])->EraseEnergyCalculation(arrayHO->wireCapCol);
										}
										sumArrayWriteEnergy += static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeEnergy;
										/* Latency for each batch write in Analog eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {	// Analog eNVM
											sumWriteLatencyAnalogPCM += maxLatencyLTP;
										}
										/* Energy consumption on array caps for eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
											if (param->writeEnergyReport) {
												if (static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) {  // 1T1R
													// The energy on selected SLs is included in WriteCell()
													sumArrayWriteEnergy += arrayHO->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage * RESETVoltage;   // Selected BL (LTP phases)
													sumArrayWriteEnergy += arrayHO->wireCapCol * RESETVoltage * RESETVoltage * (param->nHide - numBatchWriteSynapse);   // Unselected SLs (LTP phase)
													// No LTD part because all unselected rows and columns are V=0
												}
												else {
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage*RESETVoltage;    // Selected WL (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);  // Unselected WLs (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapCol *RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse);   // Unselected BLs (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);    // Unselected WLs (LTD phase)
													sumArrayWriteEnergy += arrayHO->wireCapCol * RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse); // Unselected BLs (LTD phase)
												}
											}
										}
									}
									/*Half selected Cell*/
									else { //Saturation이 안된 경우
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
											if ((!static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) && param->writeEnergyReport) { // Cross-point
												sumArrayWriteEnergy += (RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayHO->cell[j][k])->conductance * maxLatencyLTP + RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP);

											}
										}
									}
								}

							}
							//std::cout <<count3 << std::endl;
							/*if (expc == 1000) {
								std::cout << batchSize << std::endl;
							}*/
							//			/* Calculate the average number of write pulses on the selected row */
							//		#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
							//			{
							//				if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
							//					int sumNumWritePulse = 0;
							//					for (int j = 0; j < param->nHide; j++) {
							//						sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayHO->cell[j][k])->numPulse);    // Note that LTD has negative pulse number
							//					}
							//					subArrayHO->numWritePulse = sumNumWritePulse / param->nHide;
							//					double writeVoltageSquareSumRow = 0;
							//				}
							//				numWriteCellPerOperation = (double)numWriteCellPerOperation / numWriteOperationPerRow;
							//				sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayHO, numWriteOperationPerRow, numWriteCellPerOperation);
							//			}
							//			numWriteOperation += numWriteOperationPerRow;		
							//}
							arrayHO->writeEnergy += sumArrayWriteEnergy;
							subArrayHO->writeDynamicEnergy += sumNeuroSimWriteEnergy;
							numWriteOperation = numWriteOperation / param->nInput;
							subArrayHO->writeLatency += NeuroSimSubArrayWriteLatency(subArrayHO, numWriteOperation, sumWriteLatencyAnalogPCM);

							/*SET Operation*/
							/*ReWrite First Layer*/
							sumArrayWriteEnergy = 0;
							sumNeuroSimWriteEnergy = 0;
							sumWriteLatencyAnalogPCM = 0;
							numWriteOperation = 0;
							double writeVoltageLTP = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->writeVoltageLTP;
							double writePulseWidth = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->writePulseWidthLTP;
#pragma omp parallel for reduction(+:sumArrayWriteEnergy,sumNeuroSimWriteEnergy,sumWriteLatencyAnalogPCM)
							for (int k = 0; k < param->nInput; k++) {
								for (int j = 0; j < param->nHide; j++) {
									int numWriteOperationPerRow = 0;
									int numWriteCellPerOperation = 0;
									double maxLatencyLTP = 0;
									double Gp = 0;
									double Gn = 0;
									if (static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM) { //SET Saturation 된 경우 RESET operation을 진행
										double conductancePrevGp = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->conductanceGpPrev;
										double conductancePrevGn = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->conductanceGnPrev;
										static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM = false;
										//ERAESE시 weight 값 0.5
										/*std::cout << "w: " << weightGp;*/
										arrayIH->WriteCell(j, k, weight1[j][k] - 0.5, param->maxWeight, param->minWeight, false);
										//arrayIH->ReWriteCell(j, k, weight1[j][k], param->maxWeight, param->minWeight); // regular true: weight update 사용, false: 비례하여 update 
										//weight1[j][k] = arrayIH->ConductanceToWeight(j, k, param->maxWeight, param->minWeight);
										/*double w = arrayIH->ConductanceToWeight(j, k, param->maxWeight, param->minWeight);
										std::cout << j <<","<< k << ":" << w << std::endl;
										/*Gp = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductanceGp;
										Gn = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductanceGn;
										std::cout << "GP: " << Gp << "Gn: "<< Gn;*/
										numWriteCellPerOperation += 1;
										if (static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeLatencyLTP > maxLatencyLTP) {
											maxLatencyLTP = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeLatencyLTP;
										}
										if (param->writeEnergyReport) {
											static_cast<AnalogNVM*>(arrayIH->cell[j][k])->WriteEnergyCalculation(arrayIH->wireCapCol);
										}
										sumArrayWriteEnergy += static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeEnergy;
										/* Latency for each batch write in Analog eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {	// Analog eNVM
											sumWriteLatencyAnalogPCM += maxLatencyLTP;
										}
										/* Energy consumption on array caps for eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
											if (param->writeEnergyReport) {
												if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
													// The energy on selected SLs is included in WriteCell()
													sumArrayWriteEnergy += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage * RESETVoltage;   // Selected BL (LTP phases)
													sumArrayWriteEnergy += arrayIH->wireCapCol * RESETVoltage * RESETVoltage * (param->nHide - numBatchWriteSynapse);   // Unselected SLs (LTP phase)
													// No LTD part because all unselected rows and columns are V=0
												}
												else {
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage *RESETVoltage;    // Selected WL (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);  // Unselected WLs (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapCol *RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse);   // Unselected BLs (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);    // Unselected WLs (LTD phase)
													sumArrayWriteEnergy += arrayIH->wireCapCol * RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse); // Unselected BLs (LTD phase)
												}
											}
										}
									}
									/*Half selected Cell*/
									else {
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
											if ((!static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) && param->writeEnergyReport) { // Cross-point

												sumArrayWriteEnergy += (RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP + RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP);


											}
										}
									}
								}
								///* Calculate the average number of write pulses on the selected row */
								//#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
								//{
								//	if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
								//		int sumNumWritePulse = 0;
								//		for (int j = 0; j < param->nHide; j++) {
								//			sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayIH->cell[j][k])->numPulse);    // Note that LTD has negative pulse number
								//		}
								//		subArrayIH->numWritePulse = sumNumWritePulse / param->nHide;
								//		double writeVoltageSquareSumRow = 0;
								//	}
								//	numWriteCellPerOperation = (double)numWriteCellPerOperation / numWriteOperationPerRow;
								//	sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayIH, numWriteOperationPerRow, numWriteCellPerOperation);
								//}
								//numWriteOperation += numWriteOperationPerRow;

							}

							arrayIH->writeEnergy += sumArrayWriteEnergy;
							subArrayIH->writeDynamicEnergy += sumNeuroSimWriteEnergy;
							numWriteOperation = numWriteOperation / param->nInput;
							subArrayIH->writeLatency += NeuroSimSubArrayWriteLatency(subArrayIH, numWriteOperation, sumWriteLatencyAnalogPCM);

							/*SET Second Layer*/
							sumArrayWriteEnergy = 0;
							sumNeuroSimWriteEnergy = 0;
							sumWriteLatencyAnalogPCM = 0;
							numWriteOperation = 0;
							writeVoltageLTP = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->writeVoltageLTP;
							writePulseWidth = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->writePulseWidthLTP;
#pragma omp parallel for reduction(+:sumArrayWriteEnergy,sumNeuroSimWriteEnergy,sumWriteLatencyAnalogPCM)
							for (int k = 0; k < param->nHide; k++) {
								for (int j = 0; j < param->nOutput; j++) {
									int numWriteOperationPerRow = 0;
									int numWriteCellPerOperation = 0;
									double maxLatencyLTP = 0;

									if (static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM) { //SET Saturation 된 경우 RESET operation을 진행
										double conductancePrevGp = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->conductanceGpPrev;
										double conductancePrevGn = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->conductanceGnPrev;
										static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM = false;
										arrayHO->WriteCell(j, k, weight2[j][k] - 0.5, param->maxWeight, param->minWeight, false);
										//arrayHO->ReWriteCell(j, k, weight2[j][k], param->maxWeight, param->minWeight); // regular true: weight update 사용, false: 비례하여 update 
										//weight2[j][k] = arrayHO->ConductanceToWeight(j, k, param->maxWeight, param->minWeight);
										numWriteCellPerOperation += 1;
										if (static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeLatencyLTP > maxLatencyLTP) {
											maxLatencyLTP = static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeLatencyLTP;
										}
										if (param->writeEnergyReport) {
											static_cast<AnalogNVM*>(arrayHO->cell[j][k])->WriteEnergyCalculation(arrayHO->wireCapCol);
										}
										sumArrayWriteEnergy += static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeEnergy;
										/* Latency for each batch write in Analog eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {	// Analog eNVM
											sumWriteLatencyAnalogPCM += maxLatencyLTP;
										}
										/* Energy consumption on array caps for eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
											if (param->writeEnergyReport) {
												if (static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) {  // 1T1R
													// The energy on selected SLs is included in WriteCell()
													sumArrayWriteEnergy += arrayHO->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage * RESETVoltage;   // Selected BL (LTP phases)
													sumArrayWriteEnergy += arrayHO->wireCapCol * RESETVoltage * RESETVoltage * (param->nHide - numBatchWriteSynapse);   // Unselected SLs (LTP phase)
													// No LTD part because all unselected rows and columns are V=0
												}
												else {
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage *RESETVoltage;    // Selected WL (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);  // Unselected WLs (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapCol *RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse);   // Unselected BLs (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);    // Unselected WLs (LTD phase)
													sumArrayWriteEnergy += arrayHO->wireCapCol * RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse); // Unselected BLs (LTD phase)
												}
											}
										}
									}
									/*Half selected Cell*/
									else {
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
											if ((!static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) && param->writeEnergyReport) { // Cross-point

												sumArrayWriteEnergy += (RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayHO->cell[j][k])->conductance * maxLatencyLTP + RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP);


											}
										}
									}
								}
								//								/* Calculate the average number of write pulses on the selected row */
								//#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
								//								{
								//									if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
								//										int sumNumWritePulse = 0;
								//										for (int j = 0; j < param->nHide; j++) {
								//											sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayHO->cell[j][k])->numPulse);    // Note that LTD has negative pulse number
								//										}
								//										subArrayHO->numWritePulse = sumNumWritePulse / param->nHide;
								//										double writeVoltageSquareSumRow = 0;
								//									}
								//									numWriteCellPerOperation = (double)numWriteCellPerOperation / numWriteOperationPerRow;
								//									sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayHO, numWriteOperationPerRow, numWriteCellPerOperation);
								//								}
								//								numWriteOperation += numWriteOperationPerRow;


							}
							arrayHO->writeEnergy += sumArrayWriteEnergy;
							subArrayHO->writeDynamicEnergy += sumNeuroSimWriteEnergy;
							numWriteOperation = numWriteOperation / param->nInput;
							subArrayHO->writeLatency += NeuroSimSubArrayWriteLatency(subArrayHO, numWriteOperation, sumWriteLatencyAnalogPCM);

						}
						else {
							/*Read All first Layer*/
							double maxConductance = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->maxConductance;
							double ResetThr = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->ThrConductance;
							double sumArrayReadEnergy = 0; // Read Energy를 더할 임시 변수
							double readVoltage = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->readVoltage;
							double readPulseWidth = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->readPulseWidth;
						#pragma omp parallel for reduction(+: sumArrayReadEnergy)
							for (int j = 0; j < param->nHide; j++) {
								if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) { //Analog PCM
									if (static_cast<AnalogNVM*>(arrayIH->cell[0][0])->cmosAccess) { //1T1R
										sumArrayReadEnergy += arrayIH->wireCapRow*techIH.vdd*techIH.vdd*param->nInput;// All WLs open
									}
								}
								for (int n = 0; n < param->numBitInput; n++) {
									double pSumMaxAlgorithm = pow(2, n) / (param->numInputLevel - 1)*arrayIH->arrayRowSize; // numInputLevel= 2 (black or white)
									if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {
										double Isum = 0; // weight sum current
										double IsumMax = 0; //Max weight sum current
										double inputSum = 0;  // weight sum current of input vector
										for (int k = 0; k < param->nInput; k++) {
											Isum += arrayIH->ReadCell(j, k);
											if (static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductanceGp > ResetThr) {
												static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM = true;
											}
											else if (static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductanceGn > ResetThr) {
												static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM = true;
											}
											else {
												static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM = true;
											}
										}
										sumArrayReadEnergy += Isum * readVoltage*readPulseWidth; //Read energy를 다 더해줌
									}

								}
							}
							arrayIH->readEnergy += sumArrayReadEnergy;
							// Don't parallelize this loop since there may be update of member variables inside NeuroSim functions
							subArrayIH->activityRowRead = 1;
							subArrayIH->readDynamicEnergy += NeuroSimSubArrayReadEnergy(subArrayIH);
							subArrayIH->readDynamicEnergy += NeuroSimNeuronReadEnergy(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
							subArrayIH->readLatency += NeuroSimSubArrayReadLatency(subArrayIH);
							subArrayIH->readLatency += NeuroSimNeuronReadLatency(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);

							/*Read All Second Layer*/
							maxConductance = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->maxConductance;
							ResetThr = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->ThrConductance;
							sumArrayReadEnergy = 0; // Read Energy를 더할 임시 변수
							readVoltage = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->readVoltage;
							readPulseWidth = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->readPulseWidth;
						#pragma omp parallel for reduction(+: sumArrayReadEnergy)
							for (int j = 0; j < param->nOutput; j++) {
								if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) { //Analog PCM
									if (static_cast<AnalogNVM*>(arrayHO->cell[0][0])->cmosAccess) { //1T1R
										sumArrayReadEnergy += arrayHO->wireCapRow*techIH.vdd*techIH.vdd*param->nInput;// All WLs open
									}
								}
								for (int n = 0; n < param->numBitInput; n++) {
									double pSumMaxAlgorithm = pow(2, n) / (param->numInputLevel - 1)*(arrayHO->arrayRowSize); // numInputLevel= 2 (black or white)
									if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {
										double Isum = 0; // weight sum current
										double IsumMax = 0; //Max weight sum current
										double inputSum = 0;  // weight sum current of input vector
										for (int k = 0; k < param->nHide; k++) {
											//std::cout << RandNum;
											Isum += arrayHO->ReadCell(j, k);
											if (static_cast<AnalogNVM*>(arrayHO->cell[j][k])->conductanceGp > ResetThr) {
												static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM = true;
											}
											else if (static_cast<AnalogNVM*>(arrayHO->cell[j][k])->conductanceGn > ResetThr) {
												static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM = true;
											}
											else {
												static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM = true;
											}
											/*	else
												{
													static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM = false;

												}*/
										}
										sumArrayReadEnergy += Isum * readVoltage*readPulseWidth;
									}

								}
							}
							arrayHO->readEnergy += sumArrayReadEnergy;
							// Don't parallelize this loop since there may be update of member variables inside NeuroSim functions
							subArrayHO->activityRowRead = 1;
							subArrayHO->readDynamicEnergy += NeuroSimSubArrayReadEnergy(subArrayIH);
							subArrayHO->readDynamicEnergy += NeuroSimNeuronReadEnergy(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
							subArrayHO->readLatency += NeuroSimSubArrayReadLatency(subArrayIH);
							subArrayHO->readLatency += NeuroSimNeuronReadLatency(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);

							/*ERASE Opeartion*/
							/*==================Erase First Layer===================*/
							double sumArrayWriteEnergy = 0;
							double sumNeuroSimWriteEnergy = 0;
							double sumWriteLatencyAnalogPCM = 0;
							double numWriteOperation = 0;
							double RESETVoltage = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->RESETVoltage;
							double RESETPulseWidth = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->RESETPulseWidth;
#pragma omp parallel for reduction(+:sumArrayWriteEnergy,sumNeuroSimWriteEnergy,sumWriteLatencyAnalogPCM)
							for (int k = 0; k < param->nInput; k++) {
								int numWriteOperationPerRow = 0;
								int numWriteCellPerOperation = 0;
								double maxLatencyLTP = 0;
								for (int j = 0; j < param->nHide; j++) {
									if (static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM) { //SET Saturation 된 경우 RESET operation을 진행
										arrayIH->EraseCell(j, k, param->maxWeight, param->minWeight);
										numWriteCellPerOperation += 1;
										if (static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeLatencyLTP > maxLatencyLTP) {
											maxLatencyLTP = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeLatencyLTP;
										}
										if (param->writeEnergyReport) {
											static_cast<AnalogNVM*>(arrayIH->cell[j][k])->EraseEnergyCalculation(arrayIH->wireCapCol);
										}
										sumArrayWriteEnergy += static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeEnergy;
										/* Latency for each batch write in Analog eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {	// Analog eNVM
											sumWriteLatencyAnalogPCM += maxLatencyLTP;
										}
										/* Energy consumption on array caps for eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
											if (param->writeEnergyReport) {
												if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
													// The energy on selected SLs is included in WriteCell()
													sumArrayWriteEnergy += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage * RESETVoltage;   // Selected BL (LTP phases)
													sumArrayWriteEnergy += arrayIH->wireCapCol * RESETVoltage * RESETVoltage * (param->nHide - numBatchWriteSynapse);   // Unselected SLs (LTP phase)
													// No LTD part because all unselected rows and columns are V=0
												}
												else {
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage *RESETVoltage;    // Selected WL (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);  // Unselected WLs (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapCol *RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse);   // Unselected BLs (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);    // Unselected WLs (LTD phase)
													sumArrayWriteEnergy += arrayIH->wireCapCol * RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse); // Unselected BLs (LTD phase)
												}
											}
										}
									}
									/*Half selected Cell*/
									else { //Saturation이 안된경우는 그냥 남아있음.
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
											if (!static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess && param->writeEnergyReport) { // Cross-point
												sumArrayWriteEnergy += (RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP + RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP);
											}
										}
									}
								}
							}

							/* Calculate the average number of write pulses on the selected row */
				//		#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables 문제가능성 큼
				//	{
				//		if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
				//			int sumNumWritePulse = 0;
				//			for (int j = 0; j < param->nHide; j++) {
				//				sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayIH->cell[j][k])->numPulse);    // Note that LTD has negative pulse number/										}
				//				subArrayIH->numWritePulse = sumNumWritePulse / param->nHide;
				//				double writeVoltageSquareSumRow = 0;
				//			}
				//			numWriteCellPerOperation = (double)numWriteCellPerOperation / numWriteOperationPerRow;
				//			sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayIH, numWriteOperationPerRow, numWriteCellPerOperation);
				//		}
				//		numWriteOperation += numWriteOperationPerRow;
				//	}
				//}
							arrayIH->writeEnergy += sumArrayWriteEnergy;
							subArrayIH->writeDynamicEnergy += sumNeuroSimWriteEnergy;
							numWriteOperation = numWriteOperation / param->nInput;
							subArrayIH->writeLatency += NeuroSimSubArrayWriteLatency(subArrayIH, numWriteOperation, sumWriteLatencyAnalogPCM);

							/*==================Erase Second Layer===================*/
							sumArrayWriteEnergy = 0;
							sumNeuroSimWriteEnergy = 0;
							sumWriteLatencyAnalogPCM = 0;
							numWriteOperation = 0;
							RESETVoltage = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->RESETVoltage;
							RESETPulseWidth = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->RESETPulseWidth;
							/*double Gp = 0;
							double Gn = 0;*/
#pragma omp parallel for reduction(+:sumArrayWriteEnergy,sumNeuroSimWriteEnergy,sumWriteLatencyAnalogPCM)
							//int expc = 0;
							for (int k = 0; k < param->nHide; k++) {
								int numWriteOperationPerRow = 0;
								int numWriteCellPerOperation = 0;
								double maxLatencyLTP = 0;

								for (int j = 0; j < param->nOutput; j++) {

									if (static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM) { //SET Saturation 된 경우 RESET operation을 진행
										//expc += 1;
										arrayHO->EraseCell(j, k, param->maxWeight, param->minWeight);
										/*double Gp = static_cast<AnalogNVM*>(arrayHO->cell[j][k])->conductanceGp;
										double Gn = static_cast<AnalogNVM*>(arrayHO->cell[j][k])->conductanceGn;
										std::cout << "GP: " << Gp << "GN: " << Gn << std::endl;*/
										numWriteCellPerOperation += 1;
										if (static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeLatencyLTP > maxLatencyLTP) {
											maxLatencyLTP = static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeLatencyLTP;
										}
										if (param->writeEnergyReport) {
											//static_cast<AnalogNVM*>(arrayHO->cell[j][k])->EraseEnergyCalculation(arrayHO->wireCapCol);
										}
										sumArrayWriteEnergy += static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeEnergy;
										/* Latency for each batch write in Analog eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {	// Analog eNVM
											sumWriteLatencyAnalogPCM += maxLatencyLTP;
										}
										/* Energy consumption on array caps for eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
											if (param->writeEnergyReport) {
												if (static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) {  // 1T1R
													// The energy on selected SLs is included in WriteCell()
													sumArrayWriteEnergy += arrayHO->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage * RESETVoltage;   // Selected BL (LTP phases)
													sumArrayWriteEnergy += arrayHO->wireCapCol * RESETVoltage * RESETVoltage * (param->nHide - numBatchWriteSynapse);   // Unselected SLs (LTP phase)
													// No LTD part because all unselected rows and columns are V=0
												}
												else {
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage*RESETVoltage;    // Selected WL (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);  // Unselected WLs (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapCol *RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse);   // Unselected BLs (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);    // Unselected WLs (LTD phase)
													sumArrayWriteEnergy += arrayHO->wireCapCol * RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse); // Unselected BLs (LTD phase)
												}
											}
										}
									}
									/*Half selected Cell*/
									else { //Saturation이 안된 경우
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
											if ((!static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) && param->writeEnergyReport) { // Cross-point
												sumArrayWriteEnergy += (RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayHO->cell[j][k])->conductance * maxLatencyLTP + RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP);

											}
										}
									}
								}

							}
							/*if (expc == 1000) {
								std::cout << batchSize << std::endl;
							}*/
							//			/* Calculate the average number of write pulses on the selected row */
							//		#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
							//			{
							//				if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
							//					int sumNumWritePulse = 0;
							//					for (int j = 0; j < param->nHide; j++) {
							//						sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayHO->cell[j][k])->numPulse);    // Note that LTD has negative pulse number
							//					}
							//					subArrayHO->numWritePulse = sumNumWritePulse / param->nHide;
							//					double writeVoltageSquareSumRow = 0;
							//				}
							//				numWriteCellPerOperation = (double)numWriteCellPerOperation / numWriteOperationPerRow;
							//				sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayHO, numWriteOperationPerRow, numWriteCellPerOperation);
							//			}
							//			numWriteOperation += numWriteOperationPerRow;		
							//}
							arrayHO->writeEnergy += sumArrayWriteEnergy;
							subArrayHO->writeDynamicEnergy += sumNeuroSimWriteEnergy;
							numWriteOperation = numWriteOperation / param->nInput;
							subArrayHO->writeLatency += NeuroSimSubArrayWriteLatency(subArrayHO, numWriteOperation, sumWriteLatencyAnalogPCM);

							/*SET Operation*/
							/*ReWrite First Layer*/
							sumArrayWriteEnergy = 0;
							sumNeuroSimWriteEnergy = 0;
							sumWriteLatencyAnalogPCM = 0;
							numWriteOperation = 0;
							double writeVoltageLTP = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->writeVoltageLTP;
							double writePulseWidth = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->writePulseWidthLTP;
#pragma omp parallel for reduction(+:sumArrayWriteEnergy,sumNeuroSimWriteEnergy,sumWriteLatencyAnalogPCM)
							for (int k = 0; k < param->nInput; k++) {
								for (int j = 0; j < param->nHide; j++) {
									int numWriteOperationPerRow = 0;
									int numWriteCellPerOperation = 0;
									double maxLatencyLTP = 0;
									double Gp = 0;
									double Gn = 0;
									if (static_cast<AnalogNVM*>(arrayIH->cell[j][k])->SaturationPCM) { //SET Saturation 된 경우 RESET operation을 진행
										double conductancePrevGp = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->conductanceGpPrev;
										double conductancePrevGn = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->conductanceGnPrev;

										//ERAESE시 weight 값 0.5
										/*std::cout << "w: " << weightGp;*/
										arrayIH->WriteCell(j, k, weight1[j][k] - 0.5, param->maxWeight, param->minWeight, false);
										//arrayIH->ReWriteCell(j, k, weight1[j][k], param->maxWeight, param->minWeight); // regular true: weight update 사용, false: 비례하여 update 
										//weight1[j][k] = arrayIH->ConductanceToWeight(j, k, param->maxWeight, param->minWeight);
										/*double w = arrayIH->ConductanceToWeight(j, k, param->maxWeight, param->minWeight);
										std::cout << j <<","<< k << ":" << w << std::endl;
										/*Gp = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductanceGp;
										Gn = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductanceGn;
										std::cout << "GP: " << Gp << "Gn: "<< Gn;*/
										numWriteCellPerOperation += 1;
										if (static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeLatencyLTP > maxLatencyLTP) {
											maxLatencyLTP = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeLatencyLTP;
										}
										if (param->writeEnergyReport) {
											static_cast<AnalogNVM*>(arrayIH->cell[j][k])->WriteEnergyCalculation(arrayIH->wireCapCol);
										}
										sumArrayWriteEnergy += static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeEnergy;
										/* Latency for each batch write in Analog eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {	// Analog eNVM
											sumWriteLatencyAnalogPCM += maxLatencyLTP;
										}
										/* Energy consumption on array caps for eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
											if (param->writeEnergyReport) {
												if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
													// The energy on selected SLs is included in WriteCell()
													sumArrayWriteEnergy += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage * RESETVoltage;   // Selected BL (LTP phases)
													sumArrayWriteEnergy += arrayIH->wireCapCol * RESETVoltage * RESETVoltage * (param->nHide - numBatchWriteSynapse);   // Unselected SLs (LTP phase)
													// No LTD part because all unselected rows and columns are V=0
												}
												else {
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage *RESETVoltage;    // Selected WL (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);  // Unselected WLs (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapCol *RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse);   // Unselected BLs (LTP phase)
													sumArrayWriteEnergy += arrayIH->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);    // Unselected WLs (LTD phase)
													sumArrayWriteEnergy += arrayIH->wireCapCol * RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse); // Unselected BLs (LTD phase)
												}
											}
										}
									}
									/*Half selected Cell*/
									else {
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
											if ((!static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) && param->writeEnergyReport) { // Cross-point

												sumArrayWriteEnergy += (RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP + RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP);


											}
										}
									}
								}
								///* Calculate the average number of write pulses on the selected row */
								//#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
								//{
								//	if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
								//		int sumNumWritePulse = 0;
								//		for (int j = 0; j < param->nHide; j++) {
								//			sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayIH->cell[j][k])->numPulse);    // Note that LTD has negative pulse number
								//		}
								//		subArrayIH->numWritePulse = sumNumWritePulse / param->nHide;
								//		double writeVoltageSquareSumRow = 0;
								//	}
								//	numWriteCellPerOperation = (double)numWriteCellPerOperation / numWriteOperationPerRow;
								//	sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayIH, numWriteOperationPerRow, numWriteCellPerOperation);
								//}
								//numWriteOperation += numWriteOperationPerRow;

							}

							arrayIH->writeEnergy += sumArrayWriteEnergy;
							subArrayIH->writeDynamicEnergy += sumNeuroSimWriteEnergy;
							numWriteOperation = numWriteOperation / param->nInput;
							subArrayIH->writeLatency += NeuroSimSubArrayWriteLatency(subArrayIH, numWriteOperation, sumWriteLatencyAnalogPCM);

							/*SET Second Layer*/
							sumArrayWriteEnergy = 0;
							sumNeuroSimWriteEnergy = 0;
							sumWriteLatencyAnalogPCM = 0;
							numWriteOperation = 0;
							writeVoltageLTP = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->writeVoltageLTP;
							writePulseWidth = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->writePulseWidthLTP;
#pragma omp parallel for reduction(+:sumArrayWriteEnergy,sumNeuroSimWriteEnergy,sumWriteLatencyAnalogPCM)
							for (int k = 0; k < param->nHide; k++) {
								for (int j = 0; j < param->nOutput; j++) {
									int numWriteOperationPerRow = 0;
									int numWriteCellPerOperation = 0;
									double maxLatencyLTP = 0;

									if (static_cast<AnalogNVM*>(arrayHO->cell[j][k])->SaturationPCM) { //SET Saturation 된 경우 RESET operation을 진행
										double conductancePrevGp = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->conductanceGpPrev;
										double conductancePrevGn = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->conductanceGnPrev;

										arrayHO->WriteCell(j, k, weight2[j][k] - 0.5, param->maxWeight, param->minWeight, false);
										//arrayHO->ReWriteCell(j, k, weight2[j][k], param->maxWeight, param->minWeight); // regular true: weight update 사용, false: 비례하여 update 
										//weight2[j][k] = arrayHO->ConductanceToWeight(j, k, param->maxWeight, param->minWeight);
										numWriteCellPerOperation += 1;
										if (static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeLatencyLTP > maxLatencyLTP) {
											maxLatencyLTP = static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeLatencyLTP;
										}
										if (param->writeEnergyReport) {
											static_cast<AnalogNVM*>(arrayHO->cell[j][k])->WriteEnergyCalculation(arrayHO->wireCapCol);
										}
										sumArrayWriteEnergy += static_cast<AnalogNVM*>(arrayHO->cell[j][k])->writeEnergy;
										/* Latency for each batch write in Analog eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {	// Analog eNVM
											sumWriteLatencyAnalogPCM += maxLatencyLTP;
										}
										/* Energy consumption on array caps for eNVM */
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
											if (param->writeEnergyReport) {
												if (static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) {  // 1T1R
													// The energy on selected SLs is included in WriteCell()
													sumArrayWriteEnergy += arrayHO->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage * RESETVoltage;   // Selected BL (LTP phases)
													sumArrayWriteEnergy += arrayHO->wireCapCol * RESETVoltage * RESETVoltage * (param->nHide - numBatchWriteSynapse);   // Unselected SLs (LTP phase)
													// No LTD part because all unselected rows and columns are V=0
												}
												else {
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage *RESETVoltage;    // Selected WL (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);  // Unselected WLs (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapCol *RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse);   // Unselected BLs (LTP phase)
													sumArrayWriteEnergy += arrayHO->wireCapRow * RESETVoltage / 2 * RESETVoltage / 2 * (param->nInput - 1);    // Unselected WLs (LTD phase)
													sumArrayWriteEnergy += arrayHO->wireCapCol * RESETVoltage / 2 * RESETVoltage / 2 * (param->nHide - numBatchWriteSynapse); // Unselected BLs (LTD phase)
												}
											}
										}
									}
									/*Half selected Cell*/
									else {
										if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
											if ((!static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) && param->writeEnergyReport) { // Cross-point

												sumArrayWriteEnergy += (RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayHO->cell[j][k])->conductance * maxLatencyLTP + RESETVoltage / 2 * RESETVoltage / 2 * static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance * maxLatencyLTP);


											}
										}
									}
								}
								//								/* Calculate the average number of write pulses on the selected row */
								//#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
								//								{
								//									if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
								//										int sumNumWritePulse = 0;
								//										for (int j = 0; j < param->nHide; j++) {
								//											sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayHO->cell[j][k])->numPulse);    // Note that LTD has negative pulse number
								//										}
								//										subArrayHO->numWritePulse = sumNumWritePulse / param->nHide;
								//										double writeVoltageSquareSumRow = 0;
								//									}
								//									numWriteCellPerOperation = (double)numWriteCellPerOperation / numWriteOperationPerRow;
								//									sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayHO, numWriteOperationPerRow, numWriteCellPerOperation);
								//								}
								//								numWriteOperation += numWriteOperationPerRow;


							}
							arrayHO->writeEnergy += sumArrayWriteEnergy;
							subArrayHO->writeDynamicEnergy += sumNeuroSimWriteEnergy;
							numWriteOperation = numWriteOperation / param->nInput;
							subArrayHO->writeLatency += NeuroSimSubArrayWriteLatency(subArrayHO, numWriteOperation, sumWriteLatencyAnalogPCM);





						}
					}
			}
		}
	}
	}
	}

