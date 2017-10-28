/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>
#include <time.h>

#include "fann.h"

int main()
{	


	clock_t start, diff,start_big, diff_big;
	int msec;

	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 96;
	const float desired_error = (const float) 0.001;
	struct fann *ann;
	struct fann_train_data *train_data, *test_data;

	unsigned int i = 0;

	printf("Creating network.\n");

	train_data = fann_read_train_from_file("thyroid.train");

	ann = fann_create_standard(num_layers,
					  train_data->num_input, num_neurons_hidden, train_data->num_output);

	printf("Training network.\n");

	fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
	fann_set_learning_momentum(ann, 0.4f);

	fann_train_on_data(ann, train_data, 15000, 10, desired_error);
	printf("Testing network.\n");

	test_data = fann_read_train_from_file("thyroid.test");

	fann_reset_MSE(ann);
	start_big = clock();
	for(i = 0; i < fann_length_train_data(test_data); i++)
	{	//start = clock();
		fann_test(ann, test_data->input[i], test_data->output[i]);
		//diff = clock() - start;
		//msec = diff * 1000 / CLOCKS_PER_SEC;
		//printf("Test%d,in %ds and %dms\n",i,msec/1000, msec%1000);
	}
	diff_big = clock()-start_big;
	msec = diff_big * 1000 / CLOCKS_PER_SEC;
	printf("Overall testing phase,in %ds and %dms\n",msec/1000, msec%1000);
	
	printf("MSE error on test data: %f\n", fann_get_MSE(ann));

	printf("Saving network.\n");

	fann_save(ann, "robot_float.net");

	printf("Cleaning up.\n");
	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	fann_destroy(ann);

	return 0;
}
