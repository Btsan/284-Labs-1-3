#define VIENNACL_WITH_OPENMP

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>

#include <omp.h>
#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/linalg/norm_1.hpp>

using namespace std;

enum
{
	NUM_NEG = 57514,
	NUM_POS = 1731,
	NUM_DIM = 300
};

short negatives[NUM_NEG][NUM_DIM];
short positives[NUM_POS][NUM_DIM];

inline float random_weight(const float a, const float b, mt19937_64& gen) // returns Mersenne Twister random float in range [a, b]
{
	return fmod(gen()*0.0000000001, b-a)+a;
}
inline double loss(const double x, const double y) // returns logistic loss given output x and expected y
{
	return log(1.0 + exp(-y*x));
}

class Network
{
	vector<float> weights;

	public:
		Network()
		{
			weights.resize(NUM_DIM);
			random_device rd;
			mt19937_64 gen(rd());

			const float range = sqrt(2.0/300.0);
			for (unsigned int i = 0; i < 300; ++i)
			{
				weights[i] = random_weight(-range, range, gen);
			}
		}

		void train(const unsigned int iterations = 100, float step = 0.16, const float decay = 0.0016, /*const*/ unsigned int batch = 10, const bool DBG = false)
		{
			batch = NUM_NEG + NUM_POS; // ignore batch size parameter and just use full batch (Lab 01)
			if(DBG) cout << "Initialize training\n";
			random_device rd;
			mt19937 gen(rd());
			viennacl::vector<float> w;
			viennacl::matrix<float> batch_matrix(batch, 300);
			
			vector<float> tmp_ones(batch, 1);
			viennacl::vector<float> ones;
			copy(tmp_ones, ones);
			tmp_ones.clear();

			if(DBG) cout << "Fetching Parameters: "; 
			copy(weights, w);
			if(DBG) cout << w.size() << endl;

			if(DBG) cout << "Beginning iterations\n";
			const chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
			for (unsigned int it = 0; it < iterations; ++it)
			{
				if(DBG) cout << "Generating Batch: ";
				vector<short> tmp_labels(batch);
				bool neg = gen()%2;
				if (it == 0)
				for (unsigned int b = 0; b < batch; ++b)
				{
					if (b < NUM_NEG)
					{
						for (unsigned int i = 0; i < NUM_DIM; ++i)
						{
							batch_matrix(b, i) = /*(float)*/negatives[b][i];
						}
						tmp_labels[b] = 1; // only negatives of labels used
					}
					else
					{
						for (unsigned int i = 0; i < NUM_DIM; ++i)
						{
							batch_matrix(b, i) = /*(float)*/positives[b - NUM_NEG][i];
						}
						tmp_labels[b] = -1;
					}
				}
				if(DBG) cout << "rows = " << batch_matrix.size1() << "  cols = " << batch_matrix.size2() << endl;

				viennacl::vector<float> labels;
				copy(tmp_labels, labels);
				tmp_labels.clear();

				if(DBG) cout << "Calculating output: ";
				viennacl::vector<float> outputs = viennacl::linalg::prod(batch_matrix, w);
				if(DBG) cout << outputs.size() << endl;

				if(DBG) cout << "partial: ";
				viennacl::vector<float> partial = viennacl::linalg::element_exp(viennacl::linalg::element_prod(labels, outputs));
				if(DBG) cout << partial.size() << endl;

				if(DBG) cout << "Calculating loss\n";
				if(DBG) cout << "ones.size = " << ones.size() << endl;
				viennacl::vector<float> partial_1 = partial + ones;
				viennacl::vector<float> losses = viennacl::linalg::element_log(partial_1);
				float t_loss = viennacl::linalg::norm_1(losses);

				if(DBG) cout << "Calculating gradient\n";
				partial = viennacl::linalg::element_div(partial, partial_1);
				partial = viennacl::linalg::element_prod(labels, partial);
				viennacl::vector<float> gradient = viennacl::linalg::prod(trans(batch_matrix), partial);

				if(DBG) cout << "Updating parameters\n";
				w -= step*gradient;
				step -= decay;

				chrono::duration<float> elapsed = chrono::duration_cast< chrono::duration<float> >(chrono::steady_clock::now()-t0);
				cout << it << " training loss = " << t_loss << " time elapsed = " << elapsed.count() << " seconds" << endl;
			}

			copy(w.begin(), w.end(), weights.begin());

			double tot_loss = 0;
			unsigned int pos_miss = 0, neg_miss = 0;
			for (unsigned int i = 0; i < NUM_NEG; ++i)
			{
				short* input = negatives[i];
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
				short* input = positives[i];
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
};

int main(const int argc, char const *argv[])
{
	unsigned int iterations = 0;
	float step = 0.1;
	float decay = 0.01;
	unsigned int batch = 1;
	int num_threads = 0;
	string w8a_path = "../w8a_mod.txt";

	if (argc == 7)
	{
		iterations = atoi(argv[1]);
		step = atof(argv[2]);
		decay = atof(argv[3]);
		batch = atoi(argv[4]);
		num_threads = atof(argv[5]);
		w8a_path = argv[6];

		if (num_threads) omp_set_num_threads(num_threads);

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
							negatives[i][j-1] = 1.0;
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
							positives[i][j-1] = 1.0;
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
			static Network la_net = Network();
			la_net.train(iterations, step, decay, batch);
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