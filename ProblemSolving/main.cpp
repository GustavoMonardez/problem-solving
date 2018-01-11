#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

template<typename T>
void print2DVector(T& vec) {
	for (auto row : vec) {
		for (auto col : row)
			cout << col << ", ";
		cout << "\n";
	}
}
template<typename T>
void print1DVector(T& vec) {
	for (auto val : vec)
		cout << val << ",";
	cout << "\n";
}
void setMatrixZeroes(vector<vector<int>>& matrix) {
	if (matrix.empty())
		return;
	vector<vector<bool>>memo(matrix.size(), vector<bool>(matrix[0].size(), false));
	for (auto row = 0; row < matrix.size(); ++row) {
		for (auto col = 0; col < matrix[row].size(); ++col) {
			if (matrix[row][col] == 0)
				memo[row][col] = true;
		}
	}

	for (auto row = 0; row < memo.size(); ++row) {
		for (auto col = 0; col < memo[row].size(); ++col) {
			if (memo[row][col] == true) {
				//set row to zero
				for (auto i = 0; i < matrix[row].size(); ++i) {
					matrix[row][i] = 0;
				}
				//set col to zero
				for (auto i = 0; i < matrix.size(); ++i) {
					matrix[i][col] = 0;
				}
			}

		}
	}

}
void setMatrixZeroesDriver() {
	vector<vector<int>>matrix{ {1,0,1},{1,1,1},{1,1,1} };
	print2DVector(matrix);
	cout << "\n";
	setMatrixZeroes(matrix);
	print2DVector(matrix);
}
void rotateMatrix(vector<vector<int>>& matrix) {
	if (matrix.empty())
		return;
	int total_layers = matrix.size() / 2;
	int top = 0;
	int offset = 0;
	int first = 0;
	int last = 0;
	//because a layer goes around the edge of the matrix
	//top,right,bottom and left, it reduces our iterations to half
	for (int layer = 0; layer < total_layers; ++layer) {
		//moving inwards, so first element will change with
		//each iteration
		first = layer;
		//to point to last element we need to subtract the current
		//layer, because we're moving inwards and because zero base
		//we need to subtract 1
		last = matrix.size() - layer - 1;
		for (int i = first; i < last; ++i) {
			//moving inwards, so offset will be i minus first which
			//equals layer
			offset = i - first;
			//capture top-left before overriding value
			//row, in this case first, is the layer, which is why
			//we don't increase it, just the col (i)
			top = matrix[first][i];
			//left->top (element in last row(bottom) and first col)
			matrix[first][i] = matrix[last - offset][first];
			//bottom->left (last row, last col)
			matrix[last - offset][first] = matrix[last][last - offset];
			//right->bottom (current row(i) and last col
			matrix[last][last - offset] = matrix[i][last];
			//top->right
			matrix[i][last] = top;
		}
	}
	
}
void rotateMatrixDriver() {
	//vector<vector<int>>matrix{ {1,2},{3,4}};
	vector<vector<int>>matrix{ {1,2,3},{4,5,6},{7,8,9} };
	print2DVector(matrix);

	cout << "\n";
	rotateMatrix(matrix);
	print2DVector(matrix);
}
void swapWith(int num, vector<int>& A, int i) {
	int min = A[i];
	int j = i, index = i;
	for (j = i; j < A.size(); j++) {
		if (min > A[j] && A[j] > A[num]) {
			index = j;
			min = A[j];
		}
	}
	swap(A[index], A[num]);
}
void nextPermutation(vector<int>& nums) {
	int end = nums.size() - 1;
	bool found = false;
	for (auto i = end; i > 0; --i) {
		if (nums[i] > nums[i - 1]) {
			//not sure what this function is doing
			swapWith(i - 1, nums, i);
			found = true;
			break;
		}
	}
	if (!found)
		sort(nums.begin(), nums.end());
}
void nextPermutationDriver() {
	vector<int>num1{ 1,2,3 };
	vector<int>num2{ 3,2,1 };
	vector<int>num3{ 1,1,5 };
	vector<int>num4{ 20,50,113 };
	print1DVector(num1);
	print1DVector(num2);
	print1DVector(num3);
	print1DVector(num4);

	nextPermutation(num1);
	nextPermutation(num2);
	nextPermutation(num3);
	nextPermutation(num4);
	print1DVector(num1);
	print1DVector(num2);
	print1DVector(num3);
	print1DVector(num4);

}
vector<int>getPrime(int n) {
	vector<int>result;
	vector<bool>primes(n + 1, true);
	for (auto prime = 2; prime*prime <= n; ++prime) {
		if (primes[prime]) {
			for (auto i = prime * 2; i <= n; i += prime)
				primes[i] = false;
		}
	}
	for (auto prime = 2; prime <= n; ++prime) {
		if (primes[prime])
			result.push_back(prime);
	}
	return result;
}
vector<int> primesum(int n) {
	if (n < 4)
		return {};
	vector<int>prime_numbers = getPrime(n);
	vector<int>result;

	for (auto i = 0; i < prime_numbers.size(); ++i) {
		for (auto j = 0; j < prime_numbers.size(); ++j) {
			if (prime_numbers[i] + prime_numbers[j] == n) {
				result.push_back(prime_numbers[i]);
				result.push_back(prime_numbers[j]);
				//returning only one result. To return all values
				//comment out the return statement
				return result;
			}
		}
	}
	return result;
}
void primesumDriver() {
	print1DVector(primesum(4));
}
bool isPowerOfTwoInts(int n) {
	if (n <= 1)
		return true;
	for (auto x = 2; x <= sqrt(n); ++x) {
		auto power = x;
		while (power <= n) {
			power = power * x;
			if (power == n)
				return true;
		}
	}
	return false;
}
void isPowerOfTwoIntsDriver() {
	if (isPowerOfTwoInts(4))
		cout << "true\n";
	else
		cout << "false\n";
}
int uniquePaths(int A, int B) {
	if (A == 1 || B == 1)
		return 1;
	return uniquePaths(A - 1, B) + uniquePaths(A, B - 1);
}
void uniquePathsDriver() {
	cout << "uniquePaths=" << uniquePaths(2, 2) << "\n";
}
int greatestCommonDivisor(int num1, int num2) {
	if (num2 == 0)
		return num1;
	else
		return greatestCommonDivisor(num2, num1%num2);
}
void greatestCommonDivisorDriver() {
	cout << "greatestCommonDivisor=" << greatestCommonDivisor(9, 6) << "\n";
}
int reverseInteger(int num) {
	long int sol = 0;

	while (num != 0) {
		sol = sol * 10 + (num % 10);
		num = num / 10;
	}

	if (sol > INT_MAX || sol < INT_MIN) {
		return 0;
	}

	return (int)sol;
	/*string num_string = to_string(num);
	bool is_neg = false;
	if (num_string.size() > 1 && num_string[0] == '-')
		is_neg = true;

	int start = 0;
	int end = num_string.size() - 1;
	
	while (start < end) {
		swap(num_string[start], num_string[end]);
		++start;
		--end;
	}
	if (is_neg)
		return stoi(num_string) * -1;
	return stoi(num_string);*/
}
void reverseIntegerDriver() {
	cout << reverseInteger(-123) << "\n";
}
int binarySearch(const vector<int>&nums,int start,int end, int val) {
	int mid = 0;
	while (start <= end) {
		mid = (start + end) / 2;
		//cout << "nums[mid]=" << nums[mid] << " val=" << val << "\n";
		if (nums[mid] == val) {
			return mid;// cout << "inside\n";
		}
		else if (nums[mid] < val)
			start = mid + 1;
		else
			end = mid - 1;
	}
	return -1;
}
int findPivot(const vector<int>& nums) {
	int start = 0;
	int end = nums.size() - 1;
	int mid = 0;

	while (start <= end) {
		mid = (start + end) / 2;
		if (nums[mid] > nums[mid + 1])
			return mid + 1;
		else if (nums[start] < nums[mid])
			start = mid + 1;
		else
			end = mid - 1;
	}
	return -1;
}
int rotatedSortedArraySearch(const vector<int>& nums, int val) {
	int pivot = findPivot(nums);
	if (pivot == -1)
		return pivot;
	
	int right = binarySearch(nums, pivot + 1, nums.size()-1, val);
	int left =  binarySearch(nums, 0, pivot - 1, val);

	return (right != -1) ? right : left;
}
void rotatedSortedArraySearchDriver() {
	vector<int>nums{ 4, 5, 6, 7, 0, 1, 2 };
	cout << "rotatedSortedArraySearch=" << rotatedSortedArraySearch(nums, 4) << "\n";
}
int squareRoot(int num) {
	if (num == 0 || num == 1)
		return num;
	int start = 0;
	int end = num;
	int mid = (start + end) / 2;
	int answer = 0;

	while (start <= end) {
		mid = (start + end) / 2;
		if (mid*mid == num) {
			return mid;
		}
		else if (mid*mid < num) {
			start = mid + 1;
			answer = mid;
		}
		else {
			end = mid - 1;
		}
	}
	return answer;
}
void squareRootDriver() {
	cout << "squareRoot=" << squareRoot(50) << "\n";
}
vector<int> searchForRange(const vector<int>& nums, int val) {
	int pivot = binarySearch(nums, 0, nums.size() - 1, val);
	int left = binarySearch(nums, 0, pivot - 1, val);
	int right = binarySearch(nums, pivot + 1, nums.size() - 1,val);

	vector<int>result;
	if (pivot == -1)
		return { -1,-1 };

	if (left != -1) {
		result.push_back(left);
		result.push_back(pivot);
	}
	else {
		result.push_back(pivot);
		result.push_back(right);
	}

	return result;
}
void searchForRangeDriver() {
	vector<int>nums{ 5,7,7,8,8,10 };
	print1DVector(searchForRange(nums, 8));
}
int power(int base, int exponent, int mod) {
	//This function works with negative values
	if (exponent == 0) {
		return 1;
	}
	int temp = power(base, exponent / 2, mod);
	if (exponent % 2 == 0) {
		return (temp * temp) % mod;
	}
	else {
		if (exponent > 0)
			return (base * temp * temp) % mod;
		else
			return ((temp * temp) / base) % mod;
	}
}
void powerDriver() {
	cout << "power=" << power(-1, 1, 20) << "\n";
}
bool isPalindrome(string& phrase) {
	if (phrase.empty())
		return false;
	if (phrase.size() == 1)
		return true;
	int start = 0;
	int end = phrase.size() - 1;

	transform(phrase.begin(), phrase.end(),phrase.begin(), tolower);

	while (start < end) {
		if (isalnum(phrase[start]) && isalnum(phrase[end])) {
			if (phrase[start] != phrase[end])
				return false;
			++start;
			--end;
		}
		else if (isalnum(phrase[start])) {
			--end;
		}
		else if (isalnum(phrase[end])) {
			++start;
		}
	}
	return true;
}
void isPalindromeDriver() {
	string phrase1{ "A man, a plan, a canal: Panama" };
	string phrase2{ "race a car" };
	cout << phrase1 << "=" << isPalindrome(phrase1) << "\n";
	cout << phrase2 << "=" << isPalindrome(phrase2) << "\n";
}
string reverseWord(string& word) {
	int start = 0;
	int end = word.size() - 1;
	char c = ' ';

	while (start < end) {
		c = word[start];
		word[start] = word[end];
		word[end] = c;
		++start;
		--end;
	}
	return word;
}
string reverseString(string& phrase) {
	if (phrase.empty())
		return "";
	int start = 0;
	int end = phrase.size() - 1;
	string reversed_string{ "" };

	for (auto i = 0; i < phrase.size(); ++i) {
		if (phrase[i] == ' ') {
			end = i - 1;
			reversed_string = reversed_string + " " +reverseWord(phrase.substr(start, end - start+1));
			//cout << reversed_string << "\n";
			while (i < phrase.size() && phrase[i] == ' ')
				++i;
			start = i;
		}
	}
	reversed_string = reversed_string + " " + reverseWord(phrase.substr(start, phrase.size() - start+1));
	reversed_string = reverseWord(reversed_string);
	return reversed_string;
}
void reverseStringDriver() {
	string s{ "the sky is blue" };
	cout << reverseString(s) << "\n";
}
string longestPalindromicSubstring(string &s) {
	int n = s.size();
	int start = 0;
	int max_len = 1;
	vector<vector<bool>>palindromes(n,vector<bool>(n,false));
	//Single letter palindromes
	for (auto i = 0; i < n; ++i)
		palindromes[i][i] = true;
	//Two letter palindromes
	for (auto i = 0; i < n - 1; ++i) {
		if (s[i] == s[i + 1]) {
			palindromes[i][i + 1] = true;
			start = i;
			max_len = 2;
		}
	}
	//Three to n palindromes
	int j = 0;
	for (auto curr_len = 3; curr_len <= n; ++curr_len) {
		for (auto i = 0; i < n - curr_len + 1; ++i) {
			//j=last character
			j = i + curr_len - 1;
			//1.Check first and last characters
			//2. Rest of string should be palindrome
			if (s[i] == s[j] && palindromes[i + 1][j - 1]) {
				palindromes[i][j] = true;
				start = i;
				max_len = curr_len;
			}
		}
	}
	return s.substr(start, max_len + 1);
}
void longestPalindromicSubstringDriver() {
	string s{ "aaaabaaa" };
	cout << "longestPalindromicSubstring=" << longestPalindromicSubstring(s) << "\n";
}
int getRoman(char c) {
	switch (c) {
	case 'I':
		return 1;
	case 'V':
		return 5;
	case 'X':
		return 10;
	case 'L':
		return 50;
	case 'C':
		return 100;
	case 'D':
		return 500;
	case 'M':
		return 1000;
	default:
		return -1;
	}
}
int romanToInteger(string& A) {
	int result = 0;
	int s1 = 0;
	int s2 = 0;

	for (int i = 0; i < A.size(); ++i) {
		s1 = getRoman(A[i]);
		if (i + 1 < A.size()) {
			s2 = getRoman(A[i + 1]);
			if (s1 >= s2)
				result = result + s1;
			else {
				result = result + s2 - s1;
				++i;
			}
		}
		else {
			result = result + s1;
			++i;
		}
	}
	return result;
}
void romanToIntegerDriver() {
	string roman{ "XIV" };
	cout << "romanToInteger=" << romanToInteger(roman) << "\n";
}
int strstr(const string& needle, const string& haystack) {
	//find a substring ( needle ) in a string ( haystack ).
	if (needle.empty() || haystack.empty())
		return -1;
	int curr = 0;
	int count = 0;
	int temp_counter = 0;
	for (auto i = 0; i < haystack.size(); ++i) {
		if (haystack[i] == needle[curr]) {
			count = 0;
			temp_counter = i;
			for (auto j = 0; j < needle.size(); ++j) {
				if (haystack[temp_counter] == needle[j])
					++count;
				else
					--count;
				++temp_counter;
			}
			if (count == needle.size())
				return i;
		}

	}
	return -1;
}
void strstrDriver() {
	string needle{ "hello" };
	string haystack{ "world hello" };
	cout << "strstr=" << strstr(needle, haystack) << "\n";
}
bool isNumbericChar(char c) {
	return (c >= '0' && c <= '9') ? true : false;
}
int stringToInteger(string& num) {
	if (num.empty())
		return 0;
	int result = 0;
	int sign = 1;
	int i = 0;

	//if number is negative, update sign
	if (num[0] == '-') {
		sign = -1;
		++i;
	}
	for (; i < num.size(); ++i) {
		if (!isNumbericChar(num[i]))
			return 0;
		result = result * 10 + num[i] - '0';
	}
	return result * sign;
}
void stringToIntegerDriver() {
	string num{ "1028" };
	cout << "stringToInteger=" << stringToInteger(num) << "\n";
}
int singleNumberI(vector<int>& nums) {
	if (nums.empty())
		return 0;
	int result = nums[0];

	for (auto i = 1; i < nums.size(); ++i) {
		result = result ^ nums[i];
	}
	return result;
}
void singleNumberIDriver() {
	vector<int>nums{ 1, 2, 2, 3, 1 };
	cout << "singleNumberI=" << singleNumberI(nums) << "\n";
}
int singleNumberII(vector<int>& nums) {
	int ones = 0;
	int twos = 0;
	int bit_mask = 0;

	for (int i = 0; i < nums.size(); ++i) {
		twos = twos | (ones & nums[i]);
		ones = ones ^ nums[i];
		bit_mask = ~(ones & twos);
		ones = ones & bit_mask;
		twos = twos & bit_mask;
	}
	return ones;
}
void singleNumberIIDriver() {
	vector<int>nums{ 1, 2, 4, 3, 3, 2, 2, 3, 1, 1 };
	cout << "singleNumberII=" << singleNumberII(nums) << "\n";
}
unsigned int reverseBits(unsigned int num) {
	unsigned int count = sizeof(num) * 8 - 1;
	unsigned int reverse_num = num;

	num = num >> 1;
	while (num)
	{
		reverse_num = reverse_num << 1;
		reverse_num = reverse_num | num & 1;
		num = num >> 1;
		count--;
	}
	reverse_num = reverse_num  << count;
	return reverse_num;
}
void reverseBitsDriver() {
	cout << "reverseBits=" << reverseBits(3) << "\n";
}
int divideTwoInts(int dividend, int divisor) {
	long long n = dividend, m = divisor;
	// determine sign of the quotient
	int sign = n < 0 ^ m < 0 ? -1 : 1;
	// remove sign of operands
	n = abs(n), m = abs(m);
	// q stores the quotient in computation
	long long q = 0;
	// test down from the highest bit
	// accumulate the tentative value for valid bits
	for (long long t = 0, i = 31; i >= 0; i--)
		if (t + (m << i) <= n)
			t += m << i, q |= 1LL << i;
	// assign back the sign
	if (sign < 0) q = -q;
	// check for overflow and return
	return q >= INT_MAX || q < INT_MIN ? INT_MAX : q;
}
void divideTwoIntsDriver() {
	cout << "divideTwoInts=" << divideTwoInts(18, 3) << "\n";
}
int findMinXORPair(vector<int> &nums) {
	sort(nums.begin(), nums.end());
	int min_xor = INT_MAX;
	int val = 0;

	for (auto i = 0; i < nums.size()-1; ++i) {
		val = nums[i] ^ nums[i + 1];
		min_xor = min(min_xor, val);
	}
	return min_xor;
}
void findMinXORPairDriver() {
	vector<int>nums{ 0, 2, 5, 7 };
	cout << "findMinXORPair=" << findMinXORPair(nums) << "\n";
}
int removeDuplicatesFromSortedArray(vector<int>& nums) {
	if (nums.size() <= 1)
		return nums.size();

	int j = 0;
	for (auto i = 0; i < nums.size()-1; ++i) {
		//This will only copy the unique values
		//since j is not moving, the duplicate
		//will get overriden
		if (nums[i] != nums[i + 1]) {
			nums[j] = nums[i];
			++j;
		}
	}
	//copy over last value
	nums[j] = nums[nums.size() - 1];
	//increase j will get us the size, although
	//we could just return nums.size() after popping
	//duplicate value
	++j;
	nums.pop_back();
	return j;
}
void removeDuplicatesFromSortedArrayDriver() {
	vector<int>nums{ 1,1,2 };
	print1DVector(nums);
	cout << "removeDuplicatesFromSortedArray=" << removeDuplicatesFromSortedArray(nums) << "\n";
	print1DVector(nums);
}
int threeSum(vector<int>& A, int B) {
	if (A.size() <= 2)
		return -1;
	if (A.size() == 3)
		return A[0] + A[1] + A[2];

	sort(A.begin(), A.end());
	int start = 0;
	int end = A.size() - 1;
	int curr_sum = 0;
	int min_diff = INT_MAX;
	int curr_diff = 0;
	int closest_sum = 0;
	for (int i = 0; i<A.size() - 2; i++) {
		start = i + 1;
		end = A.size() - 1;
		while (start < end) {
			curr_sum = A[i] + A[start] + A[end];
			curr_diff = abs(curr_sum - B);
			if (curr_diff == 0) {
				return B;
			}
			if (curr_diff < min_diff) {
				min_diff = curr_diff;
				closest_sum = curr_sum;
			}
			if (curr_sum < B)
				++start;
			else
				--end;
		}
	}
	return closest_sum;
}
void threeSumDriver() {
	vector<int>nums{ -1,2,1,-4 };
	cout << "threeSum=" << threeSum(nums, 1) << "\n";
}
void mergeSortedArrays(vector<int>&A, vector<int>&B) {
	int index_a = A.size() - 1;
	int index_b = B.size() - 1;
	int new_size = A.size() + B.size();
	A.resize(new_size);
	--new_size;
	while (index_a >= 0 && index_b >= 0) {
		if (A[index_a] > B[index_b]) {
			A[new_size] = A[index_a];
			--index_a;
		}
		else {
			A[new_size] = B[index_b];
			--index_b;
		}
		--new_size;
	}
	while (index_a >= 0) {
		A[new_size] = A[index_a];
		--index_a;
		--new_size;
	}
	while (index_b >= 0) {
		A[new_size] = B[index_b];
		--index_b;
		--new_size;
	}
}
void mergeSortedArraysDriver() {
	vector<int>nums1{ 1,5,8 };
	vector<int>nums2{ 6,9 };
	print1DVector(nums1);
	mergeSortedArrays(nums1, nums2);
	print1DVector(nums1);
}

int main(int argc, char** argv) {
	
	//setMatrixZeroesDriver();
	//rotateMatrixDriver();
	//nextPermutationDriver();
	//primesumDriver();
	//isPowerOfTwoIntsDriver();
	//uniquePathsDriver();
	//greatestCommonDivisorDriver();
	//reverseIntegerDriver();
	//rotatedSortedArraySearchDriver();
	//squareRootDriver();
	//searchForRangeDriver();
	//powerDriver();
	//isPalindromeDriver();
	//reverseStringDriver();
	//longestPalindromicSubstringDriver();
	//romanToIntegerDriver();
	//strstrDriver();
	//stringToIntegerDriver();
	//singleNumberIDriver();
	//singleNumberIIDriver();
	//reverseBitsDriver();
	//divideTwoIntsDriver();
	//findMinXORPairDriver();
	//removeDuplicatesFromSortedArrayDriver();
	//threeSumDriver();
	mergeSortedArraysDriver();

	return 0;
}