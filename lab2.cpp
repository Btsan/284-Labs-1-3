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

short int negatives[NUM_NEG][NUM_DIM] = {0};
short int positives[NUM_POS][NUM_DIM] = {0};

inline double random_weight(const double a, const double b, mt19937_64& gen) // returns Mersenne Twister random double in range [a, b]
{
	return fmod(gen()*0.0000000001, b-a)+a;
}
inline double loss(const double x, const double y) // returns logistic loss given output x and expected y
{
	return log(1.0 + exp(-y*x));
}
double squash(const double x)
{
	return x;
	// return 0.5 + 0.5*tanh(0.5*x);
}

class Network
{
	double weights[300];

	public:
		Network()
		{
			random_device rd;
			mt19937_64 gen(rd());

			double range = sqrt(2.0/300.0);
			for (unsigned int i = 0; i < 300; ++i)
			{
				weights[i] = 0; // random_weight(-range, range, gen);
			}
		}

		void train(const unsigned int iterations = 100, double step = 0.16, const double decay = 0.0016, const unsigned int batch = 10)
		{
			short int *input;
			double output;
			random_device rd;
			mt19937 gen(rd());

			unsigned int B = (batch > 0)? batch:NUM_TOT;

			bool neg = gen()%2;
			float tloss;
			const chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

			#pragma omp parallel private(input, output) firstprivate(neg, gen)  // start threads here
			for (unsigned int it = 0; it < iterations; ++it)
			{
				if (batch) gen.seed((omp_get_thread_num()*omp_get_num_threads()*77+135790)%2864);
				tloss = 0;
				#pragma omp for schedule(static, 300) // give each pthread 300 examples
					for (unsigned int b = 0; b < B; ++b)
					{
						double expected;
						if ((batch)? neg:(b < NUM_NEG))
						{
							if (batch) input = negatives[gen()%NUM_NEG];
							else input = negatives[b];
							expected = -1.0;
						}
						else
						{
							if (batch) input = positives[gen()%NUM_POS];
							else input = positives[b - NUM_NEG];
							expected = 1.0;
						}

						output = 0;
						for (unsigned int i = 0; i < NUM_DIM; ++i) // compute output of W*X
						{
							output += input[i]*weights[i];
						}
						tloss += loss(output, expected);

						const double partial = -expected*exp(-expected*output)/(1+exp(-expected*output));

						for (unsigned int i = 0; i < NUM_DIM; ++i) // update weights
						{
							weights[i] -= step*(partial*input[i]);
						}

						if (batch) neg = gen()%2;
					}

				#pragma omp single
				{
					step -= decay;

					const chrono::duration<double> elapsed = chrono::duration_cast< chrono::duration<double> >(chrono::steady_clock::now()-t0);
					cout << it << " time elapsed = " << elapsed.count() << " seconds" << " loss = " << tloss << endl;
				}
			} // threads end here

			double avg_neg = 0, tot_loss = 0;
			unsigned int pos_miss = 0, neg_miss = 0;
			for (unsigned int i = 0; i < NUM_NEG; ++i)
			{
				input = negatives[i];
				double output = 0;
				for (unsigned int j  = 0; j < NUM_DIM; ++j)
				{
					output += input[j]*weights[j];
				}
				if (output >= 0)
				{
					++neg_miss;
				}
				tot_loss += loss(squash(output), -1.0);
				// cout << "neg #" << i << " \t" << output << " -> " << tot_loss << endl;
			}
			double avg_pos = 0;
			for (unsigned int i = 0; i < NUM_POS; ++i)
			{
				input = positives[i];
				double output = 0;
				for (unsigned int j  = 0; j < NUM_DIM; ++j)
				{
					output += input[j]*weights[j];
				}
				if (output <= 0)
				{
					++pos_miss;
				}
				tot_loss += loss(squash(output), 1.0);
				// cout << "pos #" << i << " \t" << output << " -> " << tot_loss << endl;
			}

			cout << "Total loss = " << tot_loss << "\nIncorrect on " << neg_miss << " negatives out of " << NUM_NEG << " = " << 1.0 - (double) neg_miss/NUM_NEG << "\nIncorrect on " << pos_miss << " positives out of " << NUM_POS << " = " << 1.0 - (double) pos_miss/NUM_POS << endl;

			return;
		}

		void print_weights()
		{
			for (unsigned int i = 0; i < NUM_DIM; ++i)
			{
				cout << weights[i] << ' ';
			}
			cout << endl;
		}
};

int main(const int argc, char const *argv[])
{
	unsigned int iterations = 0;
	double step = 0.1;
	double decay = 0.01;
	unsigned int batch = 1;
	unsigned int num_threads = 0;
	string w8a_path = "../w8a_mod.txt";
	if (argc == 7)
	{
		iterations = atoi(argv[1]);
		step = atof(argv[2]);
		decay = atof(argv[3]);
		batch = atoi(argv[4]);
		num_threads = atoi(argv[5]);
		w8a_path = argv[6];

		ifstream ifs;
		ifs.open(w8a_path);
		if (ifs.is_open())
		{
			string to_proc;
			for (unsigned int i = 0; i < NUM_NEG; ++i)
			{
				if (getline(ifs, to_proc))
				{
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
							negatives[i][j-1] = 1;
						}
						else
						{
							break;
						}
						next_space = to_proc.find(' ', next_colon);
					}
				}
				else
				{
					cerr << "w8a line not found: " << i << endl;
					return 1;
				}
			}
			for (unsigned int i = 0; i < NUM_POS; ++i)
			{
				if (getline(ifs, to_proc))
				{
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
							positives[i][j-1] = 1;
						}
						else
						{
							break;
						}
						next_space = to_proc.find(' ', next_colon);
					}
				}
				else
				{
					cerr << "w8a line not found: " << i << endl;
					return 1;
				}
			}
			ifs.close();

			if (num_threads) omp_set_num_threads(num_threads);

			// implement network given: iterations, step, decay, batch, negatives[][], positives[][]
			static Network hog_net = Network();
			hog_net.train(iterations, step, decay, batch);
		}
		else
		{
			cerr << "Dataset [../w8a_mod.txt] not found" << endl;
			return 1;
		}
	}
	else
	{
		cerr << "Usage: " << argv[0] << ' ' << "NUM_ITERATIONS" << ' ' << "STEP_SIZE" << ' ' << "DECAY" << ' ' << "BATCH_SIZE" << ' ' << "NUM_THREADS" << ' ' << "...w8a_mod.txt" << endl;
		return 1;
	}
	return 0;
}