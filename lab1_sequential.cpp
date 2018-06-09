#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstring>

using namespace std;

enum
{
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

inline double f(const double x)
{
	//return tanh(x);
	return x;
}
inline double df(const double x)
{
	// return 1.0 - f(x)*f(x);
	return 1.0;
}

class Network
{
	short int *input;
	double output;
	double weights[NUM_DIM];

	public:
		Network()
		{
			random_device rd;
			mt19937_64 gen(rd());

			double range = sqrt(2.0/NUM_DIM);
			for (unsigned int i = 0; i < NUM_DIM; ++i)
			{
				weights[i] = random_weight(-range, range, gen);
			}
		}

		void train(const unsigned int iterations = 100, double step = 0.16, const double decay = 0.0016, const unsigned int batch = 10, const double RATIO = 0.8)
		{
			random_device rd;
			mt19937 gen(rd());

			// training set
			vector <unsigned short> t_pos_indices;
			vector <unsigned short> t_neg_indices;
			// validation set
			vector <unsigned short> v_pos_indices;
			vector <unsigned short> v_neg_indices;

			double div = gen.max()*RATIO;
			for (unsigned int i = 0; i < NUM_NEG; ++i)
			{
				if (gen() <= div)
				{
					t_neg_indices.push_back(i);
				}
				else
				{
					v_neg_indices.push_back(i);
				}
			}
			for (unsigned int i = 0; i < NUM_POS; ++i)
			{
				if (gen() <= div)
				{
					t_pos_indices.push_back(i);
				}
				else
				{
					v_pos_indices.push_back(i);
				}
			}

			double old[NUM_DIM] = {0};

			bool neg = gen()%2;
			const chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
			for (unsigned int it = 0; it < iterations; ++it)
			{
				double gradient[NUM_DIM] = {0};
				double var_grad[NUM_DIM] = {0};
				double avg_grad[NUM_DIM] = {0};
				double t_loss = 0;
				for (unsigned int b = 0; b < batch; ++b)
				{
					unsigned short index;
					double expected;
					if (neg)
					{
						const int temp = gen()%t_neg_indices.size();
						index = t_neg_indices[temp];
						input = negatives[index];
						expected = -1.0;
					}
					else
					{
						const int temp = gen()%t_pos_indices.size();
						index = t_pos_indices[temp];
						input = positives[index];
						expected = 1.0;
					}

					output = 0;
					for (unsigned int i = 0; i < NUM_DIM; ++i)
					{
						output += input[i]*weights[i];
					}

					t_loss += loss(/*f(output)*/output, expected);

					const double partial = -expected/**df(output)*/*exp(-expected*/*f(output)*/output)/(1+exp(-expected*/*f(output)*/output));

					for (unsigned int i = 0; i < NUM_DIM; ++i)
					{
						gradient[i] += partial*input[i];
					}


					if (batch == 1)
					{
						output = 0;
						for(unsigned int i = 0; i < NUM_DIM; ++i)
						{
							output += input[i]*old[i];
						}
						const double tmp = -expected/**df(output)*/*exp(-expected*/*f(output)*/output)/(1+exp(-expected*/*f(output)*/output));
					
						for (unsigned int i = 0; i < NUM_DIM; ++i)
						{
							var_grad[i] += tmp*input[i];
							avg_grad[i] += var_grad[i];
						}

						neg = gen()%2;
						for (int i = 0; i < 1000; ++i)
						{
							if (neg)
							{
								input = negatives[t_neg_indices[gen()%t_neg_indices.size()]];
								expected = -1.0;
							}
							else
							{
								input = positives[t_pos_indices[gen()%t_pos_indices.size()]];
								expected = 1.0;
							}
							output = 0;
							for(unsigned int j = 0; j < NUM_DIM; ++j)
							{
								output += input[j]*old[j];
							}
							const double tmp_1 = -expected/**df(output)*/*exp(-expected*/*f(output)*/output)/(1+exp(-expected*/*f(output)*/output));
							for (unsigned int j = 0; j < NUM_DIM; ++j)
							{
								avg_grad[j] += tmp_1*input[j];
							}
							neg = gen()%2;
						}
						for (int i = 0; i < NUM_DIM; ++i)
						{
							avg_grad[i] /= 1001.0;
						}
					}

					neg = gen()%2;
				}

				if (batch == 1)
					for (unsigned int i = 0; i < NUM_DIM; ++i)
					{
						avg_grad[i] /= batch;
					}

				memcpy(old, weights, NUM_DIM*sizeof(double)); // save a snapshot of model at each iteration

				for (unsigned int i = 0; i < NUM_DIM; ++i)
				{
					weights[i] -= step*(gradient[i] - var_grad[i] + avg_grad[i]); // variance reduced update rule
				}

				step -= decay;

				double v_loss = 0;
				for (unsigned int i = 0; i < v_pos_indices.size(); ++i)
				{
					input = positives[v_pos_indices[i]];

					output = 0;
					for (unsigned int j = 0; j < NUM_DIM; ++j)
					{
						output += input[j]*weights[j];
					}

					v_loss += loss(/*f(output)*/output, 1.0);
				}
				for (unsigned int i = 0; i < v_neg_indices.size(); ++i)
				{
					input = negatives[v_neg_indices[i]];

					output = 0;
					for (unsigned int j = 0; j < NUM_DIM; ++j)
					{
						output += input[j]*weights[j];
					}

					v_loss += loss(/*f(output)*/output, -1.0);
				}

				chrono::duration<double> elapsed = chrono::duration_cast< chrono::duration<double> >(chrono::steady_clock::now()-t0);
				cout << it << " training loss = " << t_loss << " validation loss = " << v_loss << " time elapsed = " << elapsed.count() << " seconds" << endl;
			}

			t_pos_indices.clear();
			t_neg_indices.clear();
			v_pos_indices.clear();
			v_neg_indices.clear();

			double tot_loss = 0;
			unsigned int pos_miss = 0, neg_miss = 0;
			for (unsigned int i = 0; i < NUM_NEG; ++i)
			{
				input = negatives[i];
				double output = 0;
				for (unsigned int j  = 0; j < NUM_DIM; ++j)
				{
					output += input[j]*weights[j];
				}
				if (/*f(output)*/output >= 0)
				{
					++neg_miss;
				}
				tot_loss += loss(/*f(output)*/output, -1.0);
			}
			for (unsigned int i = 0; i < NUM_POS; ++i)
			{
				input = positives[i];
				double output = 0;
				for (unsigned int j  = 0; j < NUM_DIM; ++j)
				{
					output += input[j]*weights[j];
				}
				if (/*f(output)*/output <= 0)
				{
					++pos_miss;
				}
				tot_loss += loss(/*f(output)*/output, 1.0);
			}

			cout << "Total loss = " << tot_loss << "\nIncorrect on " << neg_miss << " negatives out of " << NUM_NEG << " = " << 1.0 - (double) neg_miss/NUM_NEG << "\nIncorrect on " << pos_miss << " positives out of " << NUM_POS << " = " << 1.0 - (double) pos_miss/NUM_POS << endl;

			return;
		}

		void print_weights()
		{
			for (unsigned int i = 0; i < 300; ++i)
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
	double ratio = 0.8;
	string w8a_path = "../w8a_mod.txt";
	if (argc == 7)
	{
		iterations = atoi(argv[1]);
		step = atof(argv[2]);
		decay = atof(argv[3]);
		batch = atoi(argv[4]);
		ratio = atof(argv[5]);
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

			// implement network given: iterations, step, decay, batch, negatives[][], positives[][]
			static Network serial_net = Network();
			serial_net.train(iterations, step, decay, batch, ratio);
		}
		else
		{
			cerr << "Dataset [../w8a_mod.txt] not found" << endl;
			return 1;
		}
	}
	else
	{
		cerr << "Usage: " << argv[0] << ' ' << "NUM_ITERATIONS" << ' ' << "STEP_SIZE" << ' ' << "DECAY" << ' ' << "BATCH_SIZE" << ' ' << "...w8a_mod.txt" << endl;
		return 1;
	}
	return 0;
}