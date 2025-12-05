#include<bits/stdc++.h>
using namespace std;

const int N = 100000;
const int NN = 10000000;
int main(){
   mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
   int n = rng() % N + 1;
	int m = rng() % N + 1;
	int k = rng() % N + 1;
	cout<<n << " "<<m<<'\n';
	for (int i = 0;i<m;++i){
		cout<<rng() % n + 1<<" ";
	}
	cout<<'\n';
	for (int i = 0;i<n;++i){
		cout<<rng() % NN + 1<<" ";		
	}
	cout<<'\n';
	cout<<k<<'\n';
	for (int i = 0;i<k;++i){
		cout<<rng() % m + 1<<" "<<rng() % m + 1<<" "<<rng()% NN + 1<<'\n';
	}
}