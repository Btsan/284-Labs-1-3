#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

enum
{
	NUM_TOT = 59245,
	NUM_NEG = 57514,
	NUM_POS = 1731,
	NUM_DIM = 300
};

short int examples[NUM_TOT][NUM_DIM] = {0};
short int labels[NUM_TOT] = {0};

inline float random_weight(const float a, const float b, mt19937_64& gen) // returns Mersenne Twister random float in range [a, b]
{
	return fmod(gen()*0.0000000001, b-a)+a;
}
inline float loss(const float x, const float y) // returns logistic loss given output x and expected y
{
	return log(1.0 + exp(-y*x));
}

class Network
{
	float weights[300];

	public:
		Network()
		{
			random_device rd;
			mt19937_64 gen(rd());

			float range = sqrt(2.0/300.0);
			for (unsigned int i = 0; i < 300; ++i)
			{
				weights[i] = random_weight(-range, range, gen);
			}
		}

		void train(const unsigned int iterations = 100, float step = 0.16, const float decay = 0.0016)
		{
			float b_grad[NUM_DIM] = {0}; // full gradient
			float output;

			float tloss = 0;
			const chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
			#pragma omp parallel private(output)  // start threads here
			for (unsigned int it = 0; it < iterations; ++it)
			{
				#pragma omp for schedule(static, 300) // give each pthread 300 examples for HOGWILD
					for (unsigned int b = 0; b < NUM_TOT; ++b)
					{
						output = 0;
						for (unsigned int i = 0; i < NUM_DIM; ++i) 
						{
							output += examples[b][i]*weights[i]; // compute output of W*X
						}
						tloss += loss(output, labels[b]);

						float grad = exp(-labels[b]*output);
						grad = -labels[b]*grad/(1+grad);

						for (unsigned int i = 0; i < NUM_DIM; ++i) // update weights
						{
							const float s_grad = examples[b][i]*grad;
							b_grad[i] += 0.6f*s_grad/NUM_TOT; // approximate full gradient with stochastic gradient
							weights[i] -= step*(0.9f*s_grad + 0.1f*b_grad[i]); // svrg update
						}
					}
				#pragma omp single
				{
					step -= decay;
					const chrono::duration<float> elapsed = chrono::duration_cast< chrono::duration<float> >(chrono::steady_clock::now()-t0);
					cout << it << " time elapsed = " << elapsed.count() << " seconds" << " loss = " << tloss << endl;
					tloss = 0;
					for (unsigned int i = 0; i < NUM_DIM; ++i)
					{
						b_grad[i] *= 0.4f; // past calculated gradient persists
					}
				}
			} // threads end here

			float tot_loss = 0;
			unsigned int pos_miss = 0, neg_miss = 0;
			for (unsigned int b = 0; b < NUM_TOT; ++b)
			{
				output = 0;
				for (unsigned int i = 0; i < NUM_DIM; ++i)
				{
					output += examples[b][i]*weights[i];
				}
				tot_loss += loss(output, labels[b]);
				if (output > 0 && labels[b] < 0) ++neg_miss;
				else if (output < 0 && labels[b] > 0) ++pos_miss;
			}

			cout << "Total loss = " << tot_loss << "\nIncorrect on " << neg_miss << " negatives out of " << NUM_NEG << " = " << 1.0 - (float) neg_miss/NUM_NEG << "\nIncorrect on " << pos_miss << " positives out of " << NUM_POS << " = " << 1.0 - (float) pos_miss/NUM_POS << endl;
			return;
		}
};

int main(const int argc, char const *argv[])
{
	unsigned int iterations = 0;
	float step = 0.1;
	float decay = 0.01;
	unsigned int num_threads = 0;
	string w8a_path = "../w8a_mod.txt";
	if (argc == 6)
	{
		iterations = atoi(argv[1]);
		step = atof(argv[2]);
		decay = atof(argv[3]);
		num_threads = atoi(argv[4]);
		w8a_path = argv[5];

		ifstream ifs;
		ifs.open(w8a_path);
		if (ifs.is_open())
		{
			string to_proc;
			for (unsigned int i = 0; i < NUM_TOT; ++i)
			{
				if (getline(ifs, to_proc))
				{
					labels[i] = (to_proc[0] == '-')? -1:1;
					int next_space, next_colon = 0;
					next_space = to_proc.find(' ', next_colon);
					while (next_space != string::npos)
					{
						next_colon = to_proc.find(':', next_space);
						if (next_colon != string::npos)
						{
							string index = to_proc.substr(next_space+1, next_colon-next_space-1);
							int j = atoi(index.c_str());
							if (j > NUM_DIM || j < 1)
							{
								cerr << "Error processing w8a line " << i << endl;
								return 1;
							}
							examples[i][j-1] = 1;
						}
						else
						{
							break;
						}
						next_space = to_proc.find(' ', next_colon);
					}
				}
			}	
			ifs.close();

			if (num_threads) omp_set_num_threads(num_threads);

			// implement network given: iterations, step, decay
			Network net = Network();
			net.train(iterations, step, decay);
		}
		else
		{
			cerr << "Dataset file not found/readable" << endl;
			return 1;
		}
	}
	else
	{
		cerr << "Usage: " << argv[0] << ' ' << "NUM_ITERATIONS" << ' ' << "STEP_SIZE" << ' ' << "DECAY" << ' ' << "NUM_THREADS" << ' ' << "DATA_FILE" << endl;
		return 1;
	}
	return 0;
}