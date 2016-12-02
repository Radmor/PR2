// Celem tego programu jest prezentacja pomiaru i analizy
//efektywnosci programu za pomoc¹  CodeAnalyst(tm).
// Implementacja mno¿enia macierzy jest realizowana za pomoca typowego
// algorytmu podrêcznikowego.
#include <stdio.h>
#include <time.h>
#include "omp.h"
#include <stdlib.h>

#define USE_MULTIPLE_THREADS true
#define MAXTHREADS 128
#define DOKL 0.000001
#define RUNS_AMOUNT 1
int NumThreads;
double start;
int blad_counter;
float blad_max;

static const int ROWS = 500;     // liczba wierszy macierzy
static const int COLUMNS = 500;  // lizba kolumn macierzy

float matrix_a[ROWS][COLUMNS];    // lewy operand
float matrix_b[ROWS][COLUMNS];    // prawy operand
float matrix_r[ROWS][COLUMNS];    // wynik
float matrix_r2[ROWS][COLUMNS];    // wynik2

FILE *result_file;

float abs(float in)
{
	return (in > 0) ? in : -in;
}

void initialize_matrices()
{
	// zdefiniowanie zawarosci poczatkowej macierzy
	//#pragma omp parallel for
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			matrix_a[i][j] = (float)rand() / RAND_MAX;
			matrix_b[i][j] = (float)rand() / RAND_MAX;
			matrix_r[i][j] = 0.0;
		}
	}
}

void initialize_matricesZ()
{
	// zdefiniowanie zawarosci poczatkowej macierzy
#pragma omp parallel for
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			matrix_r[i][j] = 0.0;
		}
	}
}

void initialize_matricesZ2()
{
	// zdefiniowanie zawarosci poczatkowej macierzy
#pragma omp parallel for
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			matrix_r2[i][j] = 0.0;
		}
	}
}
void print_result()
{
	// wydruk wyniku
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			fprintf(result_file, "%6.4f ", matrix_r[i][j]);
		}
		fprintf(result_file, "\n");
	}
}

void multiply_matrices_IJK()
{
	// mnozenie macierzy
#pragma omp parallel for
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			float sum = 0.0;
			for (int k = 0; k < COLUMNS; k++) {
				sum = sum + matrix_a[i][k] * matrix_b[k][j];
			}
			matrix_r[i][j] = sum;
		}
	}
}

void multiply_matrices_IKJ()
{
	// mnozenie macierzy
#pragma omp parallel for
	for (int i = 0; i < ROWS; i++)
		for (int k = 0; k < COLUMNS; k++)
			for (int j = 0; j < COLUMNS; j++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_JIK()
{
	// mnozenie macierzy
#pragma omp parallel for

	for (int j = 0; j < COLUMNS; j++) {
		for (int i = 0; i < ROWS; i++) {
			float sum = 0.0;
			for (int k = 0; k < COLUMNS; k++) {
				sum = sum + matrix_a[i][k] * matrix_b[k][j];
			}
			matrix_r[i][j] = sum;
		}
	}
}
void multiply_matrices_JKI()
{
	// mnozenie macierzy
#pragma omp parallel for
	for (int j = 0; j < COLUMNS; j++)
		for (int k = 0; k < COLUMNS; k++)
			for (int i = 0; i < ROWS; i++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KJI()
{
	// mnozenie macierzy
	for (int k = 0; k < COLUMNS; k++)
		for (int j = 0; j < COLUMNS; j++)
			for (int i = 0; i < ROWS; i++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KJI_omp1()
{
	// mnozenie macierzy
#pragma omp parallel for
	for (int k = 0; k < COLUMNS; k++)
		for (int j = 0; j < COLUMNS; j++)
			for (int i = 0; i < ROWS; i++)
				matrix_r2[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KJI_omp1_popraw()
{
	// mnozenie macierzy
#pragma omp parallel
{
#pragma omp for
		for (int k = 0; k < COLUMNS; k++)
			for (int j = 0; j < COLUMNS; j++)
				for (int i = 0; i < ROWS; i++)
#pragma omp atomic
					matrix_r2[i][j] += matrix_a[i][k] * matrix_b[k][j];
}
}

void multiply_matrices_KJI_omp2()
{
	// mnozenie macierzy
	for (int k = 0; k < COLUMNS; k++)
#pragma omp parallel for
		for (int j = 0; j < COLUMNS; j++)
			for (int i = 0; i < ROWS; i++)
				matrix_r2[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KJI_omp3()
{
	// mnozenie macierzy
	for (int k = 0; k < COLUMNS; k++)
		for (int j = 0; j < COLUMNS; j++)
#pragma omp parallel for
			for (int i = 0; i < ROWS; i++)
				matrix_r2[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void print_results() {
	blad_counter = blad_max = 0;
	for (int i = 0; i < ROWS; ++i)
		for (int j = 0; j < COLUMNS; ++j)
		{
			if (abs(matrix_r[i][j] - matrix_r2[i][j]) > DOKL)
			{
				blad_counter++;
				if (abs(matrix_r[i][j] - matrix_r2[i][j]) > blad_max)
					blad_max = abs(matrix_r[i][j] - matrix_r2[i][j]);
			}
			//printf("%f %f\n", matrix_r[i][j], matrix_r2[i][j]);
		}


	if (blad_counter > 0) {
		printf("LICZBA BLEDOW %i, NAJWIEKSZY BLAD  %f\n", blad_counter, blad_max);
		//fprintf(result_file, "BLEDY %i %f\n", blad_counter, blad_max);
		fprintf(result_file, "%i; %f; ", blad_counter, blad_max);
	}

}

double get_current_clock(){
    // return (double)clock() / CLK_TCK; WINDOWS
    return clock();
}


void print_elapsed_time()
{
	double elapsed;
	double resolution;

	// wyznaczenie i zapisanie czasu przetwarzania
	elapsed = get_current_clock();
    // resolution = 1.0 / CLK_TCK;
	printf("Czas: %8.4f sec \n",
		elapsed - start);

	/*fprintf(result_file,
		"Czas wykonania programu: %8.4f sec (%6.4f sec rozdzielczosc pomiaru)\n",
		elapsed - start, resolution);*/
	fprintf(result_file,
		"%8.4f; ",
		elapsed - start);
}





int main(int argc, char* argv[])
{
	//	 start = (double) clock() / CLK_TCK ;
	if ((result_file = fopen("classic.txt", "w")) == NULL) {
		fprintf(stderr, "nie mozna otworzyc pliku wyniku \n");
		perror("classic");
		return(EXIT_FAILURE);
	}


	//Determine the number of threads to use
	if (USE_MULTIPLE_THREADS) {
		// SYSTEM_INFO SysInfo;
		// GetSystemInfo(&SysInfo);
		// NumThreads = SysInfo.dwNumberOfProcessors;
        NumThreads = 4;
		if (NumThreads > MAXTHREADS)
			NumThreads = MAXTHREADS;
	}
	else
		NumThreads = 1;
	//fprintf(result_file, "Klasyczny algorytm mnozenia macierzy, liczba watkow %d \n", NumThreads);
	printf("liczba watkow  = %d\n\n", NumThreads);

	for (int h = 0;h < RUNS_AMOUNT;h++) {
		initialize_matrices();
		start = get_current_clock();
		multiply_matrices_KJI();
		printf("KJI ");
		print_elapsed_time();

		initialize_matricesZ2();
		start = get_current_clock();
		multiply_matrices_KJI_omp1();
		printf("KJI 1 ");
		print_elapsed_time();
		print_results();

		initialize_matricesZ2();
		start = get_current_clock();
		multiply_matrices_KJI_omp1_popraw();
		printf("KJI 1p");
		print_elapsed_time();
		print_results();

		initialize_matricesZ2();
		start = get_current_clock();
		multiply_matrices_KJI_omp2();
		printf("KJI 2 ");
		print_elapsed_time();
		print_results();

		initialize_matricesZ2();
		start = get_current_clock();
		multiply_matrices_KJI_omp3();
		printf("KJI 3 ");
		print_elapsed_time();
		print_results();

		fprintf(result_file, "\n");
	}

	fclose(result_file);

	return(0);
}