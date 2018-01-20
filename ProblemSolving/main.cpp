#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <stack>
#include <queue>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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
int arrayThreePointers(const vector<int>&A, const vector<int>&B, const vector<int>&C) {
	//find three closest elements from given arrays
	int min_diff = INT_MAX;
	int res_i = 0, res_j = 0, res_k = 0;
	int i = 0, j = 0, k = 0;
	int maximum = 0, minimum = 0;

	while (i < A.size() && j < B.size() && k < C.size()) {
		//find min and max of current elements
		minimum = min(A[i], min(B[j], C[k]));
		maximum = max(A[i], max(B[j], C[k]));
		//if is smallest diff, update min_diff
		if ((maximum - minimum) < min_diff) {
			res_i = i, res_j = j, res_k = k;
			min_diff = maximum - minimum;
		}
		//because we're getting absolute values,
		//the smallest value we can get is 0
		if (min_diff == 0)
			break;
		//increase the index of array with smallest
		//(minimum) value because we want the smallest
		//difference
		if (A[i] == minimum)
			++i;
		else if (B[j] == minimum)
			++j;
		else
			++k;
	}
	cout << A[res_i] << " " << B[res_j] << " " << C[res_k] << "\n";
	return max(abs(A[res_i] - B[res_j]), max(abs(B[res_j] - C[res_k]), abs(C[res_k] - A[res_i])));
}
void arrayThreePointersDriver() {
	vector<int>A{ 1,4,10 };
	vector<int>B{ 2,15,20 };
	vector<int>C{ 10,12 };
	cout << "arrayThreePointers=" << arrayThreePointers(A, B, C) << "\n";
}
struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(nullptr) {}
};
void printLinkedList(ListNode*head) {
	while (head != nullptr) {
		cout << head->val << ",";
		head = head->next;
	}
	cout << "\n";
}
ListNode* reverseListBetween(ListNode*A, int m, int n) {
	ListNode* curr = A;
	ListNode* prev = NULL;
	ListNode* temp = NULL;
	ListNode* start = NULL;
	ListNode* startTemp = NULL;
	ListNode* end = NULL;
	ListNode* endTemp = NULL;

	if (A == NULL) {
		return NULL;
	}

	int length = 0;

	while (length < m) {
		length++;
		if (length == m - 1) {
			start = curr;
		}
		else if (length == m) {
			startTemp = curr;
		}
		prev = curr;
		curr = curr->next;
	}

	while (length < n) {
		temp = curr->next;
		curr->next = prev;
		prev = curr;
		curr = temp;
		length++;
		if (length == n) {
			endTemp = prev;
			end = curr;
			startTemp->next = end;
			if (start != NULL) {
				start->next = endTemp;
			}
			else if (start == NULL) {
				A = endTemp;
			}
		}
	}

	return A;
}
void reverseListBetweenDriver() {
	ListNode* head = new ListNode(1);
	ListNode* two = new ListNode(2);
	ListNode* three = new ListNode(3);
	ListNode* four = new ListNode(4);
	ListNode* five = new ListNode(5);

	head->next = two;
	two->next = three;
	three->next = four;
	four->next = five;
	printLinkedList(head);

	head = reverseListBetween(head, 2, 4);
	printLinkedList(head);
}
ListNode* reverseLinkedList(ListNode* head) {
	if (head == nullptr)
		return nullptr;
	ListNode*prev = nullptr;
	ListNode*curr = head;
	ListNode*next = head->next;

	while (curr != nullptr) {
		next = curr->next;
		curr->next = prev;
		prev = curr;
		curr = next;
	}
	return prev;
}
ListNode* reverseLinkedListRange(ListNode* head, int start, int end) {	
	//find head
	ListNode*new_head = head;
	ListNode*prev_list = head;
	int count = 0;
	while (new_head->val != start) {
		prev_list = new_head;
		new_head = new_head->next;
		++count;
	}
	//find tail
	ListNode*new_tail = new_head;
	while (new_tail->val != end)
		new_tail = new_tail->next;
	//capture rest of list
	ListNode*rest_list = new_tail->next;
	//reverse list
	new_tail->next = nullptr;
	
	if (count == 0) {
		head = reverseLinkedList(new_head);
		new_head->next = rest_list;
	}
	else {
		prev_list->next = reverseLinkedList(new_head);
		new_head->next = rest_list;
	}

	return head;
}
void reverseLinkedListDriver() {
	ListNode* head = new ListNode(1);
	ListNode* two = new ListNode(2);
	ListNode* three = new ListNode(3);
	ListNode* four = new ListNode(4);
	ListNode* five = new ListNode(5);

	head->next = two;
	two->next = three;
	three->next = four;
	four->next = five;
	printLinkedList(head);

	head = reverseLinkedListRange(head,2,4);
	printLinkedList(head);
}
ListNode* removeDuplicates(ListNode* head) {
	if (head == nullptr || head->next == nullptr)
		return head;

	ListNode* curr = head;
	ListNode* third = nullptr;

	//first=curr, second=curr->next, third=curr->next->next
	while (curr->next != nullptr) {
		if (curr->val == curr->next->val) {
			//capture third
			third = curr->next->next;
			//delete second
			delete curr->next;
			//second is now third
			curr->next = third;
		}
		else {
			//only advance if no duplicates
			curr = curr->next;
		}
		
	}
	return head;
}
void removeDuplicatesDriver() {
	ListNode* head = new ListNode(1);
	ListNode* two = new ListNode(1);
	ListNode* three = new ListNode(2);
	ListNode* four = new ListNode(3);
	ListNode* five = new ListNode(3);

	head->next = two;
	two->next = three;
	three->next = four;
	four->next = five;
	printLinkedList(head);

	head = removeDuplicates(head);
	printLinkedList(head);
}
void insertionSort(vector<int>& nums) {
	if (nums.size() <= 1)
		return;
	int sorted_index = 0;
	int temp = 0;
	int j = 0;

	for (auto i = 1; i < nums.size(); ++i) {
		if (nums[i] < nums[sorted_index]) {
			swap(nums[i], nums[sorted_index]);
			j = sorted_index;
			while (j > 0 && nums[j] < nums[j - 1]) {				
				swap(nums[j], nums[j - 1]);				
				--j;
			}
		}
		++sorted_index;
	}
}
void insertionSortDriver() {
	vector<int>nums{ 7,4,2,5,3 };
	print1DVector(nums);
	insertionSort(nums);
	print1DVector(nums);
}
ListNode* sortedInsert(ListNode* sorted, ListNode* insert) {
	ListNode* curr = nullptr;
	if (sorted == nullptr || sorted->val >= insert->val) {
		insert->next = sorted;
		sorted = insert;
	}
	else {
		curr = sorted;
		while (curr->next != nullptr && curr->next->val < insert->val) {
			curr = curr->next;
		}
		insert->next = curr->next;
		curr->next = insert;
	}
	return sorted;
}
ListNode* insertionSortList(ListNode* head) {
	if (head == nullptr || head->next == nullptr)
		return head;

	ListNode* sorted = nullptr;
	ListNode* curr = head;
	ListNode* next = nullptr;

	while (curr != nullptr) {
		next = curr->next;
		sorted = sortedInsert(sorted, curr);
		curr = next;
	}
	return sorted;
}
void insertionSortListDriver() {
	ListNode* head = new ListNode(7);
	ListNode* two = new ListNode(2);
	ListNode* three = new ListNode(4);
	ListNode* four = new ListNode(5);
	ListNode* five = new ListNode(3);

	head->next = two;
	two->next = three;
	three->next = four;
	four->next = five;
	printLinkedList(head);

	head = insertionSortList(head);
	printLinkedList(head);
}
ListNode* detectCycle(ListNode* head) {
	//another way of keeping track of elements
	//would be to create an unordered set and when
	//i find same value, return it
	/*	if (A == NULL || A->next == NULL)
	    return NULL;
    unordered_set<ListNode*>s;
    while(A != NULL){
        if(s.find(A) != s.end())
            return A;
        s.insert(A);
        A = A->next;
    }
    return NULL;*/
	if (head == nullptr || head->next == nullptr)
		return nullptr;

	ListNode* curr = head;
	ListNode* fast = head->next;

	while (curr != nullptr && fast != nullptr && fast->next != nullptr) {
		if (curr->val == fast->val)
			return curr;
		curr = curr->next;
		fast = fast->next->next;
	}
	return nullptr;
}
void detectCycleDriver() {
	ListNode* head = new ListNode(1);
	ListNode* two = new ListNode(2);
	ListNode* three = new ListNode(3);
	ListNode* four = new ListNode(4);
	ListNode* five = new ListNode(5);

	head->next = two;
	two->next = three;
	three->next = four;
	four->next = five;
	five->next = three;
	//printLinkedList(head);

	ListNode* temp = detectCycle(head);
	if (temp == nullptr)
		cout << "no cycles\n";
	else
		cout << temp->val << "\n";
}
ListNode* addTwoNumbers(ListNode* num1, ListNode* num2) {
	ListNode* result = nullptr;
	ListNode* prev = nullptr;
	ListNode* temp = nullptr;
	int sum = 0, carry_over = 0;
	while (num1 != nullptr || num2 != nullptr) {
		//sum is carry + num1 (if there's one) + num2 (if there's one)
		sum = carry_over + (num1 ? num1->val : 0) + (num2 ? num2->val : 0);
		//set carry over to 1 if >= 10
		carry_over = (sum >= 10) ? 1 : 0;
		//get remainder
		sum = sum % 10;
		//first node, make it head(result)
		temp = new ListNode(sum);
		if (result == nullptr)
			result = temp;
		//otherwise add it to the list
		else
			prev->next = temp;

		prev = temp;
		//move only if exists
		if (num1) num1 = num1->next;
		if (num2) num2 = num2->next;
	}
	//last carry
	if (carry_over > 0)
		temp->next = new ListNode(carry_over);
	return result;
}
void addTwoNumbersDriver() {
	ListNode* num1 = new ListNode(2);
	ListNode* four = new ListNode(4);
	ListNode* three = new ListNode(3);
	num1->next = four;
	four->next = three;
	printLinkedList(num1);

	ListNode* num2 = new ListNode(5);
	ListNode* six = new ListNode(6);
	ListNode* four_2 = new ListNode(4);
	num2->next = six;
	six->next = four_2;
	printLinkedList(num2);

	ListNode* sum = addTwoNumbers(num1, num2);
	printLinkedList(sum);

}
void reverseStringStack(string& word) {
	stack<char>reverse;
	for (auto c : word)
		reverse.push(c);
	int i = 0;
	while (!reverse.empty()) {
		word[i] = reverse.top();
		reverse.pop();
		++i;
	}
}
void reverseStringStackDriver() {
	string word{ "Hello World!" };
	cout << word << "\n";
	reverseStringStack(word);
	cout << word << "\n";
}
bool validParenthesis(string& A) {
	stack<char>s;
	for (int i = 0; i < A.size(); ++i) {
		if (A[i] == '(' || A[i] == '{' || A[i] == '[') {
			if (A[i] == '(')
				s.push(')');
			if (A[i] == '{')
				s.push('}');
			if (A[i] == '[')
				s.push(']');
		}
		else {
			if (s.empty())
				return false;
			if (A[i] == s.top())
				s.pop();
			else
				return false;
		}
	}
	if (!s.empty())
		return false;
	return true;
}
void validParenthesisDriver() {
	string parens{"()[]{}"};
	string parens2{ "([)]" };
	string parens3{ ")[]{}" };

	cout << parens << "should be 1=" << validParenthesis(parens) << "\n";
	cout << parens2 << "should be 0=" << validParenthesis(parens2) << "\n";
	cout << parens3 << "should be 0=" << validParenthesis(parens3) << "\n";
}
bool redundantParens(string& expression) {
	stack<char> s;
    auto size = expression.length();
    auto i = 0;
    while(i<size)
    {
        char c = expression[i];
        if (c == '(' || c == '+' || c == '*' || c == '-' || c == '/')
            s.push(c);
        else if (c == ')')
        {
            if (s.top() == '(')
                return true;
            else
            {
                while (!s.empty() && s.top() != '(')
                    s.pop();
                s.pop();
            }
        }
        ++i;
    }
	return false;
}
void redundantParensDriver() {
	string incorrect{ "(((a+(b))+c+d))" };
	string correct{ "((a+b)+(c+d))" };
	string third{ "(((a+(b))+(c+d)))" };

	cout << "correct should be 0=" << redundantParens(correct) << "\n";
	cout << "incorrect should be 1=" << redundantParens(incorrect) << "\n";
	cout << "third should be 1=" << redundantParens(third) << "\n";
}
vector<int> slidingMaximum(const vector<int> &A, int B) {
	vector<int>result;
	//stores array elements indexes
	deque<int>q(B);
	//process first window
	int i = 0;
	for (i = 0; i < B; ++i) {
		//remove smaller elements
		while (!q.empty() && A[i] >= A[q.back()])
			q.pop_back();
		//add new element
		q.push_back(i);
	}
	//process the rest
	for (; i < A.size(); ++i) {
		//element if front is largest
		result.push_back(A[q.front()]);
		//remove elements out of current window
		while ((!q.empty()) && q.front() <= i - B)
			q.pop_front();
		//remove smaller elements
		while ((!q.empty()) && A[i] >= A[q.back()])
			q.pop_back();
		//add current element
		q.push_back(i);
	}
	//last window's element
	result.push_back(A[q.front()]);
	return result;
}
void slidingMaximumDriver() {
	vector<int>nums{ 1,3,-1,-3,5,3,6,7 };
	print1DVector(slidingMaximum(nums, 3));
}
class MinStack {
private:
	stack<int>s;
	stack<int>m;
public:
	void push(int val) {
		if (m.empty())
			m.push(val);
		else if (val < m.top())
			m.push(val);
		else if (val > m.top())
			m.push(m.top());
		s.push(val);
	}
	void pop() {
		if (!s.empty()) {
			s.pop();
			m.pop();
		}
	}
	int top() {
		if (s.empty())
			return -1;
		return s.top();
	}
	int getMin() {
		if (m.empty())
			return -1;
		return m.top();
	}
};
void minStackDriver() {
	MinStack ms;
	ms.push(1);
	ms.push(2);
	ms.push(3);
	ms.push(4);
	ms.push(5);

	cout << "top=" << ms.top() << "\n";
	cout << "getMin=" << ms.getMin() << "\n";
	ms.pop();
	cout << "top=" << ms.top() << "\n";
	cout << "getMin=" << ms.getMin() << "\n";
	ms.push(0);
	cout << "top=" << ms.top() << "\n";
	cout << "getMin=" << ms.getMin() << "\n";

}
ListNode* reverseLinkedListRecursiveHelper(ListNode* curr, ListNode* prev) {
	if (curr == nullptr)
		return prev;
	ListNode* next = curr->next;
	curr->next = prev;
	prev = curr;
	curr = next;
	return reverseLinkedListRecursiveHelper(curr, prev);
}
ListNode* reverseLinkedListRecursive(ListNode* head) {
	if (head == nullptr || head->next == nullptr)
		return head;
	return reverseLinkedListRecursiveHelper(head,nullptr);
}
void reverseLinkedListRecursiveDriver() {
	ListNode* head = new ListNode(1);
	ListNode* two = new ListNode(2);
	ListNode* three = new ListNode(3);
	ListNode* four = new ListNode(4);
	ListNode* five = new ListNode(5);

	head->next = two;
	two->next = three;
	three->next = four;
	four->next = five;
	printLinkedList(head);

	head = reverseLinkedListRecursive(head);
	printLinkedList(head);
}
void combinationsHelper(int n, int k, int curr, vector<int>& result, vector<vector<int>>& all_results) {
	if (k==0) {
		all_results.push_back(result);
		return;
	}
	if (curr == n) {
		return;
	}
	
	for (int i = curr; i < n; ++i) {
		//choose
		result.push_back(i + 1);
		//explore
		combinationsHelper(n, k-1, i+1, result, all_results);
		//unchoose
		result.pop_back();
	}
	
}
vector<vector<int>> combinations(int n, int k) {
	//n=4 -> 1-4; k= digits
	vector<vector<int>>all_results;
	vector<int>result;	
	combinationsHelper(n, k, 0, result, all_results);
	
	return all_results;
}
void combinationsDriver() {
	print2DVector(combinations(4, 2));
}
//void combinationSumHelper(vector<int> &nums, int target_sum, int curr_sum,int curr, vector<int> &result,vector<vector<int>>&all_results) {
//	if (curr_sum == target_sum) {
//		all_results.push_back(result);
//		return;
//	}
//	if (curr_sum > target_sum)
//		return;
//	for (auto i = curr; i < nums.size(); ++i) {
//		//choose
//		result.push_back(nums[i]);
//		//explore
//		combinationSumHelper(nums, target_sum, curr_sum + nums[i],curr+i, result, all_results);
//		//unchoose
//		result.pop_back();
//	}
//}
void combinationSumHelper(vector<int> &nums, int target_sum, int curr, vector<int> &result, vector<vector<int>>&all_results) {
	if (target_sum == 0) {
		all_results.push_back(result);
		return;
	}
	if (target_sum < 0 || curr == nums.size())
		return;
	
	result.push_back(nums[curr]);	
	combinationSumHelper(nums, target_sum - nums[curr], curr, result, all_results);	
	result.pop_back();
	combinationSumHelper(nums, target_sum, curr+1, result, all_results);
}
vector<vector<int>> combinationSum(vector<int> &A, int B) {
	vector<vector<int>>all_results;
	vector<int>result;
	vector<int>temp;
	sort(A.begin(), A.end());
	temp.push_back(A[0]);
	for (int i = 1; i<A.size(); ++i) {
		if (A[i] != temp[temp.size() - 1])
			temp.push_back(A[i]);
	}
	combinationSumHelper(temp, B, 0,result, all_results);
	return all_results;
}
void combinationSumDriver() {
	vector<int>nums{ 8, 10, 6, 11, 1, 16, 8 };
	int target_sum = 28;
	print2DVector(combinationSum(nums, target_sum));
}
void allPermutationsHelper(vector<int> &nums, vector<vector<int>>& all_results,int curr) {
	if (curr == nums.size()-1) {
		all_results.push_back(nums);
		return;
	}
	else {
		for (auto i = curr; i < nums.size(); ++i) {
			//choose
			swap(nums[curr], nums[i]);
			//explore
			allPermutationsHelper(nums, all_results, curr+1);
			//unchoose
			swap(nums[curr], nums[i]);
		}
	}
	
}
vector<vector<int>> allPermutations(vector<int> &A) {
	vector<vector<int>>all_results;
	allPermutationsHelper(A, all_results,0);
	return all_results;
}
void allPermutationsDriver() {
	vector<int>nums{ 1,2,3 };
	print2DVector(allPermutations(nums));
}
bool isSafe(vector<vector<string>>&board,int row, int col) {
	/*************Important things to remember************/
	//1. function gets called from column 0 to column -1, which
	//saves us from checking the current column (vertical check)
	//2. we are placing queens from left to right, top to bottom
	//for that reason we only need to check:
	//left, left-upper diagonal and left-lower diagonal
	int i = 0, j = 0;
	//check row (horizontal): placing queens from left to right so
	//there's no point checking past the passed in column since
	//we haven't placed anything yet past that point
	for (i = 0; i < col; ++i) {
		if (board[row][i] == "Q")
			return false;
	}
	//check upper left diagonal
	for (i = row, j = col; i >= 0 && j >= 0; --i, --j) {
		if (board[i][j] == "Q")
			return false;
	}
	//check lower left diagonal
	for (i = row, j = col; i < board.size() && j >= 0; ++i, --j) {
		if (board[i][j] == "Q")
			return false;
	}
	return true;
}
void solveNQueensHelper(int total_queens,int col, vector<vector<string>>&board, vector<string>& config, vector<vector<string>>&solutions) {
	if (col >= total_queens) {
		//all queens placed
		solutions.push_back(config);
		return;
	}
	else {
		for (auto row = 0; row < total_queens; ++row) {
			if (isSafe(board,row, col)) {
				//choose
				board[row][col] = "Q";
				config[row][col] = 'Q';
				//explore
				solveNQueensHelper(total_queens, col + 1, board, config,solutions);
				//unchoose
				board[row][col] = ".";
				config[row][col] = '.';
			}
		}
	}
}
vector<vector<string>> solveNQueens(int A) {
	vector<vector<string>>solutions;
	//there's no possible solutions
	if (A == 2 || A == 3)
		return solutions;
	//if board size is 1 we can place one queen
	if (A == 1)
		return { {"Q"} };
	vector<vector<string>>board(A,vector<string>(A,"."));
	vector<string>config(A,"....");
	solveNQueensHelper(A, 0, board, config, solutions);
	return solutions;
}
void solveNQueensDriver() {
	print2DVector(solveNQueens(4));
}
bool isPalindrome(const string& s, int start, int end) {
	while (start < end) {
		if (s[start] != s[end])
			return false;
		++start;
		--end;
	}
	return true;
}
void palindromePartitioningHelper(string s, int start,int end,vector<string>palindrome,vector<vector<string>>& all_palindromes) {
	if (start >= end) {
		all_palindromes.push_back(palindrome);
		return;
	}
	else {
		for (auto i = start; i < end; ++i) {
			if (isPalindrome(s, start, i)) {
				//choose
				palindrome.push_back(s.substr(start, i - start + 1));
				//explore
				palindromePartitioningHelper(s, i + 1, end, palindrome, all_palindromes);
				//unchoose
				palindrome.pop_back();
			}
		}
	}
}
vector<vector<string>> palindromePartitioning(string& A) {
	vector<vector<string>>all_palindromes;
	vector<string>palindrome;
	palindromePartitioningHelper(A, 0, A.size(), palindrome, all_palindromes);
	return all_palindromes;
}
void palindromePartitioningDriver() {
	string s{ "aab" };
	print2DVector(palindromePartitioning(s));
}
void generateParenthesisHelper(int n,int pos, int open, int close,string& parens, vector<string>&all_parens) {
	if (close == n) {
		all_parens.push_back(parens);
		return;
	}
	else {
		if (open > close) {
			parens[pos] = ')';
			generateParenthesisHelper(n, pos + 1, open, close + 1,parens,all_parens);
		}
		if (open < n) {
			parens[pos] = '(';
			generateParenthesisHelper(n, pos + 1, open + 1, close, parens, all_parens);
		}
	}
}
vector<string> generateParenthesis(int A) {
	vector<string>all_parens;
	string parens(A * 2,' ');
	generateParenthesisHelper(A, 0, 0, 0, parens, all_parens);
	sort(all_parens.begin(), all_parens.end());
	return all_parens;
}
void generateParenthesisDriver() {
	print1DVector(generateParenthesis(3));
}
vector<int> twoSum(vector<int>& nums, int target_sum) {
	vector<int>result;
	unordered_map<int, int>table;

	for (auto i = 0; i < nums.size(); ++i) {
		if (table.find(target_sum - nums[i]) != table.end()) {
			result.push_back(table[target_sum - nums[i]]);
			result.push_back(i + 1);
			return result;
		}
		if (table.find(nums[i]) == table.end())
			table[nums[i]] = i + 1;
	}
	return result;
}
void twoSumDriver() {
	vector<int>nums{ 2, 7, 11, 15 };
	int target_sum = 9;
	print1DVector(twoSum(nums, target_sum));
}
vector<int> findSubstring(string A, const vector<string> &B) {
	vector<int>result;
	int wsize = B[0].size();
	int lsize = B.size();

	if (A.size() == 0 || B.size() == 0)
		return result;

	unordered_map<string, int> myMap;

	for (int i = 0; i < B.size(); i++) {
		if (myMap.find(B[i]) != myMap.end())
			myMap[B[i]]++;
		else
			myMap[B[i]] = 1;
	}

	int i = 0;

	while ((i + wsize*lsize - 1) < A.size()) {
		unordered_map<string, int> tempMap;
		int j = 0;
		while (j < A.size()) {
			string temp = A.substr(i + j*wsize, wsize);
			if (myMap.find(temp) == myMap.end()) {
				break;
			}
			else {
				if (tempMap.find(temp) == tempMap.end()) {
					tempMap[temp] = 1;
				}
				else {
					tempMap[temp]++;
				}
				if (tempMap[temp] > myMap[temp]) {
					break;
				}
				j++;
			}
			if (j == lsize) {
				result.push_back(i);
			}
		}
		i++;
	}
	return result;

}
void findSubstringDriver() {

}
int lengthOfLongestSubstring(string A) {
	unordered_set<char>s;
	int max_len = 0;
	for (auto c : A) {
		if (s.find(c) != s.end()) {
			max_len = max(max_len, int(s.size()));
			s.clear();
		}
		else {
			s.insert(c);
		}
	}
	return max_len;
	/*unordered_map<char,int>hash;
	int curr_len = 0;
	int max_len = 1;
	int i = 0;

	while(i < A.size()) {
		if (hash.find(A[i]) == hash.end()) {
			hash[A[i]] = i;
			++curr_len;
			++i;
		}
		else {
			i = hash[A[i]] + 1;
			hash.clear();
			max_len = max(max_len, curr_len);
			curr_len = 0;
		}
		
	}
	return max(max_len,curr_len);*/
}
void lengthOfLongestSubstringDriver() {
	string s{ "bbbbb" };
	cout << "lengthOfLongestSubstring=" << lengthOfLongestSubstring(s) << "\n";
}
bool usedInRow(vector<vector<char>>& grid, int row, int num) {
	for (auto col = 0; col < grid[row].size(); ++col) {
		if (grid[row][col] == num)
			return true;
	}
	return false;
}
bool usedInCol(vector<vector<char>>& grid, int col, int num) {
	for (auto row = 0; row < grid.size(); ++row) {
		if (grid[row][col] == num)
			return true;
	}
	return false;
}
bool usedInBox(vector<vector<char>>& grid, int box_start_row,int box_start_col, int num) {
	int box_size = 3;
	for (auto row = 0; row < box_size; ++row) {
		for (auto col = 0; col < box_size; ++col) {
			if (grid[row+box_start_row][col+box_start_col] == num)
				return true;
		}
	}
	return false;
}
bool isSudokuSafe(vector<vector<char>>& grid, int row, int col, int num) {
	//row - row%3 gets us the first row of the particular box. same for cols.
	return !usedInRow(grid, row, num) &&
		   !usedInCol(grid, col, num) &&
		   !usedInBox(grid, row - row % 3, col - col % 3, num);
}
bool emptyGridLocation(vector<vector<char>>& grid, int &row, int &col) {
	for (row = 0; row < grid.size(); ++row) {
		for (col = 0; col < grid[row].size(); ++col) {
			if (grid[row][col] == '.')
				return true;
		}
	}
	return false;
}
bool solveSudokuHelper(vector<vector<char>> &grid) {
	int row = 0, col = 0;
	//if no more empty spots
	if (!emptyGridLocation(grid, row, col))
		return true;
	for (auto num = 1; num <= 9; ++num) {
		if (isSudokuSafe(grid, row, col, num)) {
			//choose
			grid[row][col] = num + '0'; cout << num + '0' << "\n";
			//explore
			if (solveSudokuHelper(grid))
				return true;
			//unchoose
			grid[row][col] = '.';
		}
	}
	return false;
}
void solveSudoku(vector<vector<char>> &A) {
	if (solveSudokuHelper(A)) {
		cout << "success\n";
		print2DVector(A);
	}
	else
		cout << "something went wrong\n";
}
void solveSudokuDriver() {
	//vector<vector<char>>grid{{"53..7...."},{"6..195..."},{".98....6."},{"8...6...3"},{"4..8.3..1"},{"7...2...6"},{".6....28."},{"...419..5"},{"....8..79"} };
	//vector<vector<char>>grid{ { '5'},{'3'}, {'.'}, {'.'}, {'7'},{ '.' },{ '.' },{ '.' },{ '.' } };
	vector<vector<char>>grid(9, vector<char>(9, '.'));
	solveSudoku(grid);
}
int maxPoints(vector<int> &A, vector<int> &B) {
	unordered_map<double, int>map;
	int max_points = 0;
	int duplicate = 1;
	int vertical = 0;

	for (auto i = 0; i < A.size(); ++i) {
		duplicate = 1;
		vertical = 0;
		for (auto j = i + 1; j < A.size(); ++j) {
			if (A[i] == A[j]) {
				if (A[i] == B[j])
					++duplicate;
				else
					++vertical;
			}
			else {
				double slope = 0.0;
				double x = A[j] - A[i];
				double y = B[j] - B[i];
				if (B[j] != B[i])
					slope = (1.0 * (y / x));
				if (map.find(slope) != map.end())
					++map[slope];
				else
					map[slope] = 1;
			}
		}
		auto it = map.begin();
		while (it != map.end()) {
			int t = it->second;
			if ((t + duplicate) > max_points) {
				max_points = t + duplicate;
			}
			it++;
		}
		if ((vertical + duplicate) > max_points) {
			max_points = vertical + duplicate;
		}
		map.clear();
	}
	return max_points;
}
void maxPointsDriver() {
	vector<int>x{ 1,1 };
	vector<int>y{ 2,2 };
	cout << "maxPoints=" << maxPoints(x, y) << "\n";
}
vector<vector<int>> findAnagrams(const vector<string> &A) {
	vector<vector<int>>result;
	vector<string>row;
	unordered_map<string, vector<int>>anagram;
	//sort letters of each word and copy them to an
	//aux container (row)
	for (auto i = 0; i < A.size(); ++i) {
		string temp{};
		temp.append(A[i]);
		sort(temp.begin(), temp.end());
		row.push_back(temp);
	}
	//for each word, push indexes where they're found
	//because maps don't accept duplicates it will push
	//the each index of the same word into the same vector
	for (auto i = 0; i < A.size(); ++i) {
		anagram[row[i]].push_back(i + 1);
	}
	//final step is to copy over those vectors containing indexes
	//into our result 2D vector
	for (auto it = anagram.begin(); it != anagram.end(); ++it) {
		result.push_back(it->second);
	}

	return result;
}
void findAnagramsDriver() {
	vector<string>words{ {"cat"}, {"dog"},{"god"}, {"tca"} };
	print2DVector(findAnagrams(words));
}
string minWindow(string& s, string& pattern) {
	if (s.size() < pattern.size())
		return "";

	unordered_map<char, int>strings{};
	unordered_map<char, int>all_patterns{};

	//store characters and their occurrence
	for (auto i = 0; i < pattern.size(); ++i)
		++all_patterns[pattern[i]];
	
	int start = 0, start_index = -1, min_len = INT_MAX, count=0;

	for (auto i = 0; i < s.size(); ++i) {
		//store characters and their occurrence
		++strings[s[i]];
		//if we find match increase count
		if (all_patterns[s[i]] != 0 && strings[s[i]] <= all_patterns[s[i]])
			++count;
		//we found window
		if (count == pattern.size()) {
			//if some characters occur more often than patterns, then
			//remove them to get smaller window
			while (strings[s[start]] > all_patterns[s[start]] || all_patterns[s[start]] == 0) {
				if (strings[s[start]] > all_patterns[s[start]])
					--strings[s[start]];
				++start;
			}
			//update window size
			int window_len = i - start + 1;
			if (min_len > window_len) {
				min_len = window_len;
				start_index = start;
			}
		}
	}
	if (start_index == -1)
		return "";
	return s.substr(start_index, min_len);
}
void minWindowDriver() {
	string s{ "ADOBECODEBANC" };
	string pattern{ "ABC" };
	cout << minWindow(s, pattern) << "\n";
}
ListNode* mergeKSortedLists(vector<ListNode*> &lists) {
	vector<int>values;
	for (auto i = 0; i < lists.size(); ++i) {
		ListNode* curr = lists[i];
		while (curr) {
			values.push_back(curr->val);
			curr = curr->next;
		}
	}
	sort(values.begin(), values.end());
	ListNode* new_list = new ListNode(values[0]);
	ListNode* prev = nullptr;
	for (auto i = 1; i < values.size(); ++i) {
		ListNode* temp = new ListNode(values[i]);
		if (new_list->next == nullptr) {
			new_list->next = temp;
			prev = temp;
		}
		else {
			prev->next = temp;
			prev = temp;
		}
	}
	return new_list;
	/* multiset<int>values;
	ListNode *curr = NULL;
	for (auto list : A) {
		curr = list;
		while (curr != NULL) {
			values.insert(curr->val);
			curr = curr->next;
		}
	}
	if (!values.empty()) {
		ListNode *merge_list = NULL;
		ListNode *temp = NULL;
		curr = NULL;
		for (auto it = values.begin(); it != values.end(); ++it) {
			temp = new ListNode(*it);
			if (merge_list == NULL) {
				merge_list = temp;
				merge_list->next = NULL;
			}
			else if (merge_list->next == NULL) {
				curr = temp;
				merge_list->next = curr;
			}
			else {
				curr->next = temp;
				curr = temp;
			}
		}
		return merge_list;
	}
	return NULL;*/
}
void mergeKSortedListsDriver() {
	//list one
	ListNode* list_one = new ListNode(1);
	ListNode* three = new ListNode(3);
	ListNode* five = new ListNode(5);
	ListNode* seven = new ListNode(7);
	list_one->next = three;
	three->next = five;
	five->next = seven;
	printLinkedList(list_one);
	//list two
	ListNode* list_two = new ListNode(2);
	ListNode* four = new ListNode(4);
	ListNode* six = new ListNode(6);
	ListNode* eight = new ListNode(8);
	list_two->next = four;
	four->next = six;
	six->next = eight;
	printLinkedList(list_two);
	//list three
	ListNode* list_three = new ListNode(0);
	ListNode* nine = new ListNode(9);
	ListNode* ten = new ListNode(10);
	ListNode* eleven = new ListNode(11);
	list_three->next = nine;
	nine->next = ten;
	ten->next = eleven;
	printLinkedList(list_three);
	//merge sorted lits
	//vector<ListNode*>lists{ list_one,list_two,list_three };
	ListNode* head = new ListNode(1);
	ListNode* next = new ListNode(6);
	head->next = next;
	vector<ListNode*>lists{ head,nullptr,nullptr };
	ListNode* sorted_list = mergeKSortedLists(lists);
	printLinkedList(sorted_list);
}
class LeastRecentlyUsed {
private:
	size_t capacity;
	size_t count;
	unordered_map<int, int>map;
	unordered_map<int, int>freq;
	int findLeastUsedKey() {
		int min_val = INT_MAX;
		int key = 0;
		for (auto it = freq.begin(); it != freq.end(); ++it) {
			cout << it->first << " " << it->second << "\n";
			if (it->second < min_val) {
				min_val = it->second;
				key = it->first;
			}
		}
		return key;
	}
	void decreaseAll(int key) {
		for (auto it = freq.begin(); it != freq.end(); ++it) {
			if (it->first != key)
				--freq[it->first];
		}
	}
public:
	LeastRecentlyUsed(size_t capacity) :capacity(capacity),count(0) {}
	void set(int key, int val) {
		if(count >= capacity) {
			//remove LRU
			int lru_key = findLeastUsedKey();
			cout << "erasing=" << lru_key << "\n";
			map.erase(lru_key);
			freq.erase(lru_key);
			--count;
		}
		map[key] = val;
		++freq[key];
		++count;
		//decreaseAll(key);
	}
	int get(int key) {
		if (map.find(key) == map.end())
			return -1;
		++freq[key];
		decreaseAll(key);
		return map[key];
	}
};
void LeastRecentlyUsedDriver() {
	LeastRecentlyUsed LRU(2);
	LRU.set(1, 10);
	LRU.set(5, 12);
	cout << LRU.get(5) << "\n";
	cout << LRU.get(1) << "\n";
	cout << LRU.get(10) << "=-1\n";
	LRU.set(6, 14);
	cout << LRU.get(5) << "\n";
}
vector<int> distinctNumbers(vector<int> &A, int B) {
	//geeks for geeks efficient solution with explantion
	vector<int>result;
	map<int, int>map;
	int count = 0;
	//first window
	for (auto i = 0; i < B; ++i) {
		if (map[A[i]] == 0)
			++count;
		++map[A[i]];
	}
	result.push_back(count);
	//remaining array
	for (auto i = B; i < A.size(); ++i) {
		if (map[A[i - B]] == 1)
			--count;
		map[A[i - B]] = map[A[i - B]] - 1;

		if (map[A[i]] == 0)
			++count;
		map[A[i]] = map[A[i]] + 1;

		result.push_back(count);
	}
	return result;
	/*vector<int>result;
	unordered_set<int>set;
	for (auto i = 0; i < A.size()-B+1; ++i) {
		for (auto j = i; j < B+i; ++j) {
			set.insert(A[j]);
		}
		result.push_back(set.size());
		set.clear();
	}
	return result;*/
}
void distinctNumbersDriver() {
	vector<int>nums{ 1, 2, 1, 3, 4, 3 };
	int k = 3;
	print1DVector(distinctNumbers(nums, k));
}
struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
int isValidBSTHelper(TreeNode* curr) {
	if (curr == NULL)
		return 1;
	if (curr->left == NULL && curr->right == NULL)
		return 1;
	if (curr->left == NULL || curr->right == NULL)
		return 0;
	return curr->val > curr->left->val &&
		curr->val < curr->right->val &&
		isValidBSTHelper(curr->left) &&
		isValidBSTHelper(curr->right);
}
/*int checkValid(TreeNode* root, int min, int max){
    if(root == NULL){
        return 1;
    }
    else if((root->val < max) && 
        (root->val > min) && 
        (checkValid(root->left, min, root->val)) &&
        (checkValid(root->right, root->val, max))){
        return 1;        
    }
    return 0;
}
int Solution::isValidBST(TreeNode* A) {
return checkValid(A, INT_MIN, INT_MAX);
}
*/
int isValidBST(TreeNode* A) {
	return  isValidBSTHelper(A);
}
void isValidBSTDriver() {
	TreeNode* root1 = new TreeNode(1);
	TreeNode* two = new TreeNode(2);
	TreeNode* three = new TreeNode(3);
	root1->left = two;
	root1->right = three;
	cout << "root1=" << isValidBST(root1) << "\n";
	TreeNode* root2 = new TreeNode(2);
	TreeNode* left = new TreeNode(1);
	TreeNode* right = new TreeNode(3);
	root2->left = left;
	root2->right = right;
	cout << "root2=" << isValidBST(root2) << "\n";
	
}
void inOrderRecursive(TreeNode* root) {
	if (root == NULL)
		return;
	inOrderRecursive(root->left);
	cout << root->val << ",";
	inOrderRecursive(root->right);
}
vector<int> inorderIterative(TreeNode* A) {
	vector<int>inorder;
	stack<TreeNode*>s;
	TreeNode* curr = A;

	while (true) {
		if (curr != NULL) {
			s.push(curr);
			curr = curr->left;
		}
		else {
			if (!s.empty()) {
				curr = s.top(); s.pop();
				inorder.push_back(curr->val);
				curr = curr->right;
			}
			else {
				break;
			}

		}
	}
	return inorder;
}
void inOrderDriver() {
	TreeNode* root1 = new TreeNode(1);
	TreeNode* two = new TreeNode(2);
	TreeNode* three = new TreeNode(3);
	root1->right = two;
	two->left = three;
	inOrderRecursive(root1);
	cout << "\n";
	print1DVector(inorderIterative(root1));

}
vector<int> preorderIterative(TreeNode* A) {
	if (A == NULL)
		return {};
	vector<int>preorder;
	stack<TreeNode*>s;
	TreeNode* curr = NULL;

	s.push(A);
	while (!s.empty()) {
		//1. push curr
		preorder.push_back(s.top()->val);
		curr = s.top(); s.pop();
		//2.push right first because left needs to come out first
		if (curr->right != NULL)
			s.push(curr->right);
		//3.push left last so we can pop before right
		if (curr->left != NULL)
			s.push(curr->left);
	}
	return preorder;
}
void preOrderDriver() {
	TreeNode* root1 = new TreeNode(1);
	TreeNode* two = new TreeNode(2);
	TreeNode* three = new TreeNode(3);
	root1->right = two;
	two->left = three;
	print1DVector(preorderIterative(root1));
}
vector<int> postorderIterative(TreeNode* A) {
	if (A == NULL)
		return {};
	vector<int>postorder;
	stack<TreeNode*>s1;
	stack<TreeNode*>s2;
	TreeNode* curr = NULL;
	
	s1.push(A);

	while (!s1.empty()) {
		//1.parent node goes first, because is the last one to come out
		curr = s1.top(); s1.pop();
		s2.push(curr);
		//in reverse order, left has to come after right to be first
		if (curr->left != NULL)
			s1.push(curr->left);
		//right will become top in stack and second in reverse order to
		//be popped off after parent
		if (curr->right != NULL)
			s1.push(curr->right);
	}
	while (!s2.empty()) {
		postorder.push_back(s2.top()->val); s2.pop();
	}
	return postorder;
}
void postOrderDriver() {
	TreeNode* root1 = new TreeNode(1);
	TreeNode* two = new TreeNode(2);
	TreeNode* three = new TreeNode(3);
	root1->right = two;
	two->left = three;
	print1DVector(postorderIterative(root1));
}
struct TreeLinkNode {
	int val;
	TreeLinkNode *left, *right, *next;
	TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
};
void nextRightPointers(TreeLinkNode* A) {
	if (A == NULL)
		return;
	TreeLinkNode* prev = NULL;
	TreeLinkNode* curr = NULL;
	queue<TreeLinkNode*>q;
	queue<int>level;
	q.push(A);
	level.push(0);

	int curr_level = 0;
	int prev_level = 0;

	while (!q.empty()) {
		curr_level = level.front();
		curr = q.front();
		//transitioning to next level, so if the last element (prev)
		//on the prev element is not null, it means is the element
		//all the way to the right and it needs to point to NULL
		if (curr_level != prev_level) {
			if (prev != NULL)
				prev->next = NULL;
		}
		//otherwise, prev(node on the left) needs to point to curr
		//(which is the node to its right)  if not null of course
		else {
			if (prev != NULL)
				prev->next = curr;
		}
		//push nodes from next level
		if (curr->left != NULL) {
			q.push(curr->left);
			level.push(curr_level + 1);
		}
		if (curr->right != NULL) {
			q.push(curr->right);
			level.push(curr_level + 1);
		}
		//move to next level
		prev_level = curr_level;
		prev = curr;
		q.pop();
		level.pop();
	}
	//take care of last element if any
	if (prev != NULL)
		prev->next = NULL;
}
void nextRightPointersDriver() {
	TreeLinkNode* root = new TreeLinkNode(1);
	TreeLinkNode* two = new TreeLinkNode(2);
	TreeLinkNode* three = new TreeLinkNode(3);
	TreeLinkNode* four = new TreeLinkNode(4);
	TreeLinkNode* five = new TreeLinkNode(5);
	TreeLinkNode* six = new TreeLinkNode(6);
	TreeLinkNode* seven = new TreeLinkNode(7);

	root->left = two;
	root->right = three;
	two->left = four;
	two->right = five;
	three->left = six;
	three->right = seven;

	nextRightPointers(root);
	if(five->next == NULL)
		cout << "root->next->NULL\n";
	else
		cout << "root->next->" << five->next->val << "\n";
}
int hasPathSumHelper(TreeNode* curr, int sum) {
	if (curr == NULL)
		return 0;
	if (curr->left == NULL && curr->right == NULL) {
		//another way to think about it is, if I were to subtract the curr node's value
		//from sum and the result is zero, then we found a path or like the approach below
		//if the sum, that we've been subtracting values from, equals to the current node's
		//value it means we found a path
		if (sum == curr->val)
			return 1;
	}
	return hasPathSumHelper(curr->left, sum - curr->val) || hasPathSumHelper(curr->right, sum - curr->val);
}
int hasPathSum(TreeNode* A, int B) {
	return hasPathSumHelper(A, B);
}
void hasPathSumDriver() {
	TreeNode* root = new TreeNode(5);
	TreeNode* four = new TreeNode(4);
	TreeNode* eight = new TreeNode(8);
	TreeNode* eleven = new TreeNode(11);
	TreeNode* thirteen = new TreeNode(13);
	TreeNode* four_again = new TreeNode(4);
	TreeNode* seven = new TreeNode(7);
	TreeNode* two = new TreeNode(2);
	TreeNode* one = new TreeNode(1);

	root->left = four;
	root->right = eight;
	four->left = eleven;
	eleven->left = seven;
	eleven->right = two;
	eight->left = thirteen;
	eight->right = four_again;
	four_again->right = one;

	cout << "hasPathSum=" << hasPathSum(root, 22) << "\n";
}
int maxDepthOfBST(TreeNode* root) {
	if (root == NULL)
		return 0;
	return 1 + max(maxDepthOfBST(root->left), maxDepthOfBST(root->right));
}
void maxDepthOfBSTDriver() {
	TreeNode* root = new TreeNode(5);
	TreeNode* four = new TreeNode(4);
	TreeNode* eight = new TreeNode(8);
	TreeNode* eleven = new TreeNode(11);
	TreeNode* thirteen = new TreeNode(13);
	TreeNode* four_again = new TreeNode(4);
	TreeNode* seven = new TreeNode(7);
	TreeNode* two = new TreeNode(2);
	TreeNode* one = new TreeNode(1);

	root->left = four;
	root->right = eight;
	four->left = eleven;
	eleven->left = seven;
	eleven->right = two;
	eight->left = thirteen;
	eight->right = four_again;
	four_again->right = one;

	cout << "maxDepthOfBST=" << maxDepthOfBST(root) << "\n";
}
int isBalancedBST(TreeNode* root) {
	if (root == NULL)
		return 1;
	int left = maxDepthOfBST(root->left);
	int right = maxDepthOfBST(root->right);
	return abs(left - right) <= 1 &&
		isBalancedBST(root->left) &&
		isBalancedBST(root->right);
}
void isBalancedBSTDriver() {

}
int identicalThrees(TreeNode* t1, TreeNode* t2) {
	if (t1 == NULL && t2 == NULL)
		return 1;
	if (t1 == NULL || t2 == NULL)
		return 0;
	int left = identicalThrees(t1->left, t2->left);
	int right = identicalThrees(t1->right, t2->right);
	return (t1->val == t2->val) && left && right;
}
void identicalThreesDriver() {
	TreeNode* t1 = new TreeNode(1);
	TreeNode* t1_two = new TreeNode(2);
	TreeNode* t1_three = new TreeNode(3);
	t1->left = t1_two;
	t1->right = t1_three;

	TreeNode* t2 = new TreeNode(1);
	TreeNode* t2_two = new TreeNode(3);
	TreeNode* t2_three = new TreeNode(2);
	t2->left = t2_two;
	t2->right = t2_three;

	cout << "identicalThrees=" << identicalThrees(t1, t2) << '\n';
}
int isSymmetricHelper(TreeNode* t1 ,TreeNode* t2) {
	if (t1 == NULL && t2 == NULL)
		return 1;
	return (t1 != NULL && t2 != NULL) &&
				 (t1->val == t2->val) &&
		isSymmetricHelper(t1->left, t2->right) &&
		isSymmetricHelper(t1->right, t2->left);
}
int isSymmetric(TreeNode* A) {
	return isSymmetricHelper(A, A);
}
void isSymmetricDriver() {

}
bool findTreeNode(TreeNode* curr,int val) {
	if (curr == NULL)
		return false;
	if (curr->val == val)
		return true;
	return findTreeNode(curr->left, val) || findTreeNode(curr->right, val);
}
TreeNode* leastCommonAncestorHelper(TreeNode* curr, int val1, int val2) {
	if (curr == NULL)
		return NULL;
	if (curr->val == val1 || curr->val == val2)
		return curr;
	TreeNode* left = leastCommonAncestorHelper(curr->left, val1, val2);
	TreeNode* right = leastCommonAncestorHelper(curr->right, val1, val2);
	//found ancestor
	if (left != NULL && right != NULL)
		return curr;
	//did not find on left, but both values might be on
	//right subtree or vicesversa
	if (left == NULL)
		return right;
	else
		return left;
}
int leastCommonAncestor(TreeNode* root, int val1, int val2) {
	bool n1 = findTreeNode(root, val1);
	bool n2 = findTreeNode(root, val2);

	if (n1 == false || n2 == false)
		return -1;
	TreeNode* ancestor = leastCommonAncestorHelper(root, val1, val2);
	return ancestor == NULL ? -1 : ancestor->val;
}
void leastCommonAncestorDriver() {
	TreeNode* root = new TreeNode(3);
	TreeNode* five = new TreeNode(5);
	TreeNode* one = new TreeNode(1);
	TreeNode* six = new TreeNode(6);
	TreeNode* two = new TreeNode(2);
	TreeNode* seven = new TreeNode(7);
	TreeNode* four = new TreeNode(4);
	TreeNode* zero = new TreeNode(0);
	TreeNode* eight = new TreeNode(8);

	root->left = five;
	root->right = one;
	five->left = six;
	five->right = two;
	two->left = seven;
	two->right = four;
	one->left = zero;
	one->right = eight;

	cout << "leastCommonAncestor=" << leastCommonAncestor(root, 5, 4) << '\n';
}
struct Trie {
	Trie* edges[26];
	int words;
	Trie() {
		for (int i = 0; i<26; i++)
			edges[i] = 0;
		words = 0;
	}
};
void addToTrie(Trie *head, string s) {
	int n = s.length();
	Trie *current = head;
	int i;
	for (i = 0; i<n; i++) {
		current->words += 1;
		if (!current->edges[s[i] - 'a'])
			current->edges[s[i] - 'a'] = new Trie();
		current = current->edges[s[i] - 'a'];
	}
}
string findPrefix(string s, Trie *head) {
	string pfx = "";
	Trie *current = head;
	int i = 0, n = s.length();
	current = current->edges[s[i] - 'a'];
	pfx += s[i];
	for (i = 1; i<n; i++) {
		if (current->words == 1)
			return pfx;
		current = current->edges[s[i] - 'a'];
		pfx += s[i];
	}
	return pfx;

}
vector<string> shortestUniquePrefix(vector<string> &A) {
	vector<string>prefixes;
	Trie* head = new Trie();
	for (auto i = 0; i < A.size(); ++i)
		addToTrie(head, A[i]);
	for (auto i = 0; i < A.size(); ++i)
		prefixes.push_back(findPrefix(A[i],head));
	return prefixes;
}
void shortestUniquePrefixDriver() {
	vector<string>words{ {"zebra"},{"dog"},{"duck"},{"dove"} };
	print1DVector(words);
	print1DVector(shortestUniquePrefix(words));
}
TreeNode* flattenBinaryTreeToList(TreeNode* A) {
	/*if (curr == NULL)
		return;
	if (curr->left != NULL && curr->right == NULL)
		swap(curr->left, curr->right);
	else if (curr->left != NULL && curr->right != NULL) {
		swap(curr->left, curr->right);
		TreeNode *temp = curr->right;
		while (temp->right != NULL)
			temp = temp->right;
		temp->right = curr->left;
		curr->left = NULL;
	}
	flattenHelper(curr->right);*/
	if (A == NULL)
		return NULL;

	TreeNode*curr = A;
	while (curr != NULL) {
		//attach the right subtree to the right most leaf on the left subtree
		if (curr->left != NULL) {
			TreeNode* right_most = curr->left;
			while (right_most->right != NULL) {
				right_most = right_most->right;
			}
			//attaching
			right_most->right = curr->right;

			curr->right = curr->left;
			curr->left = NULL;
		}
		//flatten the rest of the tree
		curr = curr->right;
	}
}
void flattenBinaryTreeToListDriver() {

}
void printBinaryTree(TreeNode* curr) {
	if (curr == NULL)
		return;
	queue<TreeNode*>q;
	q.push(curr);

	while (!q.empty()) {
		curr = q.front(); q.pop();
		cout << curr->val << ",";
		if (curr->left != NULL)
			q.push(curr->left);
		if (curr->right != NULL)
			q.push(curr->right);
	}
	cout << "\n";
}
int findInOrderIndex(vector<int>& nums, int start, int end, int val) {
	for (auto i = start; i <= end; ++i) {
		if (nums[i] == val)
			return i;
	}
	return -1;
}
TreeNode* buildTreePreorderInorderHelper(vector<int> &preorder, vector<int> &inorder,int start, int end) {
	static int preorder_index = 0;
	if (start > end)
		return NULL;
	TreeNode* node = new TreeNode(preorder[preorder_index]);
	++preorder_index;
	//if no children
	if (start == end)
		return node;
	//else find node in inorder
	int inorder_index = findInOrderIndex(preorder, start, end, node->val);
	//construct tree
	node->left = buildTreePreorderInorderHelper(preorder, inorder, start, inorder_index - 1);
	node->right = buildTreePreorderInorderHelper(preorder, inorder, inorder_index+1, end);
	return node;
}
TreeNode* buildTreePreorderInorder(vector<int> &preorder, vector<int> &inorder) {
	return buildTreePreorderInorderHelper(preorder, inorder, 0, inorder.size() - 1);
}
void buildTreePreorderInorderDriver() {
	vector<int>preorder{ 1,2,3 };
	vector<int>inorder{ 2,1,3 };
	//buildTreePreorderInorder(preorder, inorder);
	printBinaryTree(buildTreePreorderInorder(preorder, inorder));
}
int sizeTree(TreeNode *root) {
	if (root == NULL)
		return 0;

	return 1 + sizeTree(root->left) + sizeTree(root->right);
}
int kthsmallest(TreeNode* root, int k) {
	int leftSize = sizeTree(root->left);

	if (leftSize == k - 1)
		return root->val;

	if (leftSize > k - 1)
		return kthsmallest(root->left, k);

	return kthsmallest(root->right, k - leftSize - 1);
}
void kthsmallestDriver() {
	TreeNode* root = new TreeNode(2);
	TreeNode* one = new TreeNode(1);
	TreeNode* three = new TreeNode(3);
	root->left = one;
	root->right = three;
	//printBinaryTree(root);
	cout << "kthsmallest=" << kthsmallest(root, 2) << "\n";
}
class BSTIterator {
private:
	TreeNode* root;
	deque<int>inorder_values;
	void inorder(TreeNode* curr) {
		if (curr == NULL)
			return;
		inorder(curr->left);
		inorder_values.push_back(curr->val);
		inorder(curr->right);
	}
public:
	BSTIterator(TreeNode* root) :root(root) {
		inorder(root);
	}
	int next() {
		int next_val = inorder_values.front(); inorder_values.pop_front();
		return next_val;
	}
	bool hasNext() {
		return inorder_values.empty() ? false : true;
	}
};
void BSTIteratorDriver() {
	TreeNode* root = new TreeNode(8);
	TreeNode* three = new TreeNode(3);
	TreeNode* ten = new TreeNode(10);
	TreeNode* one = new TreeNode(1);
	TreeNode* six = new TreeNode(6);
	TreeNode* fourteen = new TreeNode(14);
	TreeNode* four = new TreeNode(4);
	TreeNode* seven = new TreeNode(7);
	TreeNode* thirteen = new TreeNode(13);
	root->left = three;
	root->right = ten;
	three->left = one;
	three->right = six;
	six->left = four;
	six->right = seven;
	ten->right = fourteen;
	fourteen->left = thirteen;
	printBinaryTree(root);

	BSTIterator it(root);
	cout << it.next() << "\n";
	cout << it.next() << "\n";
	cout << it.next() << "\n";
	cout << it.next() << "\n";
	cout << it.next() << "\n";
	cout << it.next() << "\n";
	cout << it.next() << "\n";
	cout << it.next() << "\n";
	cout << it.hasNext() << "\n";
	cout << it.next() << "\n";
	cout << it.hasNext() << "\n";
}
int climbStairsRecursive(int total_steps) {
	if (total_steps <= 2)
		return total_steps;
	return climbStairsRecursive(total_steps - 1) + climbStairsRecursive(total_steps - 2);
}
int climbStairsTopDown(int total_steps,vector<int>& memo) {
	if (total_steps <= 2)
		return total_steps;
	if (memo[total_steps] == 0) {
		memo[total_steps] = climbStairsTopDown(total_steps - 1,memo) + climbStairsTopDown(total_steps - 2,memo);
	}
	return memo[total_steps];
}
int climbStairsBottomUp(int total_steps) {
	vector<int>memo(total_steps + 1, 0);
	memo[0] = 1;
	memo[1] = 1;

	for (auto i = 2; i <= total_steps; ++i) {
		memo[i] = memo[i - 1] + memo[i - 2];
	}
	return memo[total_steps];
}
int climbStairsAnyNumOfSteps(int total_steps, int k_steps, vector<int>& memo) {
	if (total_steps <= 1)
		return 1;
	if (memo[total_steps] == 0) {
		for (auto i = 1; i <= k_steps && total_steps - i >= 0; ++i) {
			memo[total_steps] = memo[total_steps] + climbStairsAnyNumOfSteps(total_steps - i, k_steps, memo);
		}
	}
	return memo[total_steps];
}
void climbStairsDriver() {
	cout << "climbStairsRecursive=" << climbStairsRecursive(3) << "\n";
	vector<int>memo1(11, 0);
	cout << "climbStairsTopDown=" << climbStairsTopDown(10,memo1) << "\n";
	
	cout << "climbStairsBottomUp=" << climbStairsBottomUp(10) << "\n";

	vector<int>memo2(11, 0);
	cout << "climbStairsAnyNumOfSteps=" << climbStairsAnyNumOfSteps(10,3, memo2) << "\n";
}
int editDistanceRecursive(string A, string B, int a_size, int b_size) {
	if (a_size == 0)
		return b_size;
	if (b_size == 0)
		return a_size;
	//last chars are the same, keep going
	if (A[a_size - 1] == B[b_size - 1])
		return editDistanceRecursive(A, B, a_size - 1, b_size - 1);
	//if not the same, consider all edits: insert, delete and replace
	return 1 + min(editDistanceRecursive(A, B, a_size, b_size - 1),//insert
		min(editDistanceRecursive(A, B, a_size - 1, b_size), //delete
			editDistanceRecursive(A, B, a_size - 1, b_size - 1)));//replace
}
int editDistanceBottomUp(string A, string B, int a_size, int b_size) {
	vector<vector<int>>memo(a_size + 1, vector<int>(b_size + 1));
	for (auto i = 0; i <= a_size; ++i) {
		for (auto j = 0; j <= b_size; ++j) {
			//first string empty
			if (i == 0)
				memo[i][j] = j;
			//second string empty
			else if (j == 0)
				memo[i][j] = i;
			//last chars are the same
			else if (A[i - 1] == B[j - 1])
				memo[i][j] = memo[i - 1][j - 1];
			//consider all edits: insert, remove and replace
			else
				memo[i][j] = 1 + min(memo[i][j - 1], min(memo[i - 1][j], memo[i - 1][j - 1]));
			
		}
	}
	return memo[a_size][b_size];
}
void editDistanceDriver() {
	string s1{ "Anshuman" };
	string s2{ "Antihuman" };
	cout << "editDistanceRecursive=" << editDistanceRecursive(s1, s2, s1.size(), s2.size()) << '\n';
	cout << "editDistanceTopDown=" << editDistanceBottomUp(s1, s2, s1.size(), s2.size()) << '\n';
}
int longestIncreasingSubsequenceRecursive(vector<int>& arr, int n, int *max_ref){
	if (n == 1)
		return 1;
	// 'max_ending_here' is length of LIS ending with arr[n-1]
	int res, max_ending_here = 1;
	/* Recursively get all LIS ending with arr[0], arr[1] ...
	arr[n-2]. If   arr[i-1] is smaller than arr[n-1], and
	max ending with arr[n-1] needs to be updated, then
	update it */
	for (int i = 1; i < n; i++)
	{
		res = longestIncreasingSubsequenceRecursive(arr, i, max_ref);
		if (arr[i - 1] < arr[n - 1] && res + 1 > max_ending_here)
			max_ending_here = res + 1;
	}

	// Compare max_ending_here with the overall max. And
	// update the overall max if needed
	if (*max_ref < max_ending_here)
		*max_ref = max_ending_here;

	// Return length of LIS ending with arr[n-1]
	return max_ending_here;
}
int longestIncreasingSubsequenceNLogN(vector<int>& nums) {
	vector<int>parent(nums.size());
	vector<int>increasingSubsequence(nums.size() + 1);
	int len = 0,start=0,mid=0,end=0,pos=0;

	for (auto i = 0; i < nums.size(); ++i) {
		start = 1;
		end = len;
		while (start <= end) {
			mid = (start + end) / 2;
			if (nums[increasingSubsequence[mid]] < nums[i])
				start = mid + 1;
			else
				end = mid - 1;
		}
		pos = start;
		parent[i] = increasingSubsequence[pos - 1];
		increasingSubsequence[pos] = i;
		if (pos > len)
			len = pos;
	}
	return len;
}
void longestIncreasingSubsequenceDriver() {
	vector<int>nums{ 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15 };
	int max = 1;
	cout << "longestIncreasingSubsequenceRecursive=" << longestIncreasingSubsequenceRecursive(nums, nums.size(), &max) << '\n';
	cout << "longestIncreasingSubsequenceNLogN=" << longestIncreasingSubsequenceNLogN(nums) << '\n';
}
int canJumpRecursive(vector<int> &nums, int start, int end) {
	//destination same as source
	if (start == end)
		return 0;
	//nothing reachable
	if (nums[start] == 0)
		return INT_MAX;
	//traverse all reachable points
	int min = INT_MAX;
	int jumps = 0;
	for (auto i = start + 1; i <= end && i <= start + nums[start]; ++i) {
		jumps = canJumpRecursive(nums, i, end);
		if (jumps != INT_MAX && jumps + 1 < min)
			min = jumps + 1;
	}
	return min;
}
int canJumpN(vector<int>& nums) {
	if (nums.size() <= 1)
		return 1;
	if (nums[0] == 0)
		return 0;

	int max_reach = nums[0];
	int step = nums[0];
	int jump = 1;
	
	for (auto i = 1; i < nums.size(); ++i) {
		//made it to the end
		if (i == nums.size() - 1)
			return 1;
		max_reach = max(max_reach, i + nums[i]);
		--step;
		if (step == 0) {
			++jump;
			if (i >= max_reach)
				return 0;
			step = max_reach - i;
		}
	}
	return 0;
}
void canJumpDriver() {
	vector<int>nums1{ 2,3,1,1,4 };
	vector<int>nums2{ 3,2,1,0,4 };
	cout << "canJumpRecursive(1)=" << canJumpRecursive(nums1,0,nums1.size()-1) << '\n';
	if(canJumpRecursive(nums2, 0, nums2.size() - 1) == INT_MAX)
		cout << "nums2(0)\n" ;
	cout << "canJumpN(1)=" << canJumpN(nums1) << '\n';
	cout << "canJumpN(0)=" << canJumpN(nums2) << '\n';
}
int maxRectangleInMatrix(vector<vector<int>>& matrix) {
	//does not work for all cases
	if (matrix.empty())
		return 0;
	int rows = matrix.size();
	int cols = matrix[0].size();
	vector<vector<int>>aux(rows, vector<int>(cols));
	//set first row
	for (auto i = 0; i < cols; ++i)
		aux[0][i] = matrix[0][i];
	//set first col
	for (auto i = 0; i < rows; ++i)
		aux[i][0] = matrix[i][0];
	//set rest
	for (auto i = 1; i < rows; ++i) {
		for (auto j = 1; j < cols; ++j) {
			if (matrix[i][j] == 1)
				aux[i][j] = min(aux[i][j - 1], min(aux[i - 1][j], aux[i - 1][j - 1])) + 1;
			else
				aux[i][j] = 0;
		}
	}
	//find max
	int max_rect = aux[0][0];
	for (auto i = 0; i < rows; ++i) {
		for (auto j = 0; j < rows; ++j) {
			max_rect = max(max_rect, aux[i][j]);
		}
	}
	return max_rect * max_rect;
}
void maxRectangleInMatrixDriver() {
	vector<vector<int>>matrix{ {1,1,1},{0,1,1},{1,0,0} };
	cout << "maxRectangleInMatrix=" << maxRectangleInMatrix(matrix) << '\n';
}
vector<int> rodCut(int A, vector<int> &B) {
	return {};
}
void rodCutDriver() {

}
int nDigitNumsWithSumRecursiveHelper(int digits, int sum) {
	//if no digits and sum is zero then there would
	//technically be one way of getting 0
	if (digits == 0)
		return sum == 0;
	//only one way to get zero
	if (sum == 0)
		return 1;
	//traverse all
	long long int answer = 0;
	for (auto i = 0; i <= 9; ++i) {
		if (sum - i >= 0)
			answer = answer + nDigitNumsWithSumRecursiveHelper(digits - 1, sum - i);
	}
	return answer;
}
int nDigitNumsWithSumRecursive(int digits, int sum) {
	long long int answer = 0;
	for (auto i = 1; i <= 9; ++i) {
		if (sum - i >= 0)
			answer = answer + nDigitNumsWithSumRecursiveHelper(digits - 1, sum - i);
	}
	return answer;
}
int nDigitNumsWithSumTopDownHelper(int digits, int sum, vector<vector<int>>& memo) {
	if (digits == 0)
		return sum == 0;
	if (memo[digits][sum] == 0) {
		long long int answer = 0;
		for (auto i = 0; i <= 9; ++i) {
			if (sum - i >= 0)
				answer = answer + nDigitNumsWithSumTopDownHelper(digits - 1, sum - i, memo);
		}
		//mod it for really large inputs
		memo[digits][sum] = answer % 1000000007;
	}
	return memo[digits][sum];
}
int nDigitNumsWithSumTopDown(int digits, int sum) {
	//wrapper function to take care of leading zeroes
	vector<vector<int>>memo(digits + 1, vector<int>(sum + 1, 0));
	long long int answer = 0;
	for (auto i = 1; i <= 9; ++i) {
		if (sum - i >= 0)
			answer = answer + nDigitNumsWithSumTopDownHelper(digits - 1, sum - i, memo);
	}
	//mod it for really large inputs
	return answer % 1000000007;
}
void nDigitNumsWithSumDriver() {
	cout << "nDigitNumsWithSum=" << nDigitNumsWithSumRecursive(3, 5) << '\n';
	cout << "nDigitNumsWithSumTopDown=" << nDigitNumsWithSumTopDown(3, 5) << '\n';
}
int longestValidParenthesesBottomUp(string& word) {
	vector<int>memo(word.size(), 0);
	int len = 0;
	int index = 0;
	for (auto i = 1; i < word.size(); ++i) {
		if (word[i] == ')') {
			if (word[i - 1] == '(') {
				memo[i] = 2;
				if (i >= 2)
					memo[i] = memo[i] + memo[i - 2];
			}
			else {
				index = i - memo[i - 1] - 1;
				if (index >= 0 && word[index] == '(') {
					memo[i] = memo[i - 1] + 2;
					if (index > 0)
						memo[i] = memo[i] + memo[index - 1];
				}
			}
		}
		len = max(len, memo[i]);
	}
	return len;
}
void longestValidParenthesesBottomUpDriver() {
	string s{ "(()" };
	cout << "longestValidParenthesesBottomUp=" << longestValidParenthesesBottomUp(s) << '\n';
}
int maxProfit(const vector<int> &A) {
	if (A.empty())
		return 0;
	int investment = A[0];
	int profit = 0;
	for (auto i = 1; i < A.size(); ++i) {
		if (investment < A[i])
			profit = profit + A[i] - investment;
		investment = A[i];
	}
	return profit;
}
void maxProfitDriver() {
	vector<int>nums{ 7551982, 8124939, 4023780, 7868369, 4412570, 2542961, 7380261, 1164290, 7781065, 1164599, 2563492, 5354415, 4994454, 2627136, 5933501, 668219, 1821804, 7818378, 33654, 4167689, 8652323, 5750640, 9822437, 3466466, 554089, 6168826, 335687, 2466661, 8511732, 6288553, 2905889, 7747975, 3744045, 1545003, 1008624, 8041203, 7176125, 4321092, 714053, 7200073, 166697, 7814651, 3090485, 8318668, 6600364, 3352620, 2430137, 7685821, 1442555, 828955, 6540266, 5305436, 116568, 1883410, 7975347, 9629015, 4735259, 6559041, 1832532, 5840170, 6983732, 5886179, 1496505, 7241412, 144558, 9462840, 8579314, 2488436, 9677478, 7589124, 5636642, 2440601, 1767332, 2399786, 6299635, 8534665, 1367339, 805592, 5572668, 6990026, 8465261, 4808596, 7641452, 8464860, 3170126, 7403200, 6932907, 3776122, 1313688, 3992189, 2382116, 3886952, 349816, 1596435, 7353742, 9964868, 9882224, 3818546, 3885458, 1200559, 3910256, 7949895, 463872, 6392805, 9513226, 3427933, 3470571, 6225817, 552452, 5567651, 6414423, 6701681, 4725847, 894529, 8046603, 426263, 5280891, 9197661, 9764507, 1740413, 9530261, 9163599, 7561587, 5848442, 7312422, 4794268, 5793465, 5039382, 5147388, 7346933, 4697363, 6436473, 5159752, 2207985, 8256403, 8958858, 6099618, 2172252, 3063342, 4324166, 3919237, 8985768, 2703255, 2386343, 3064166, 247762, 7271683, 1812487, 7163753, 4635382, 449426, 2561592, 3746615, 8741199, 6696192, 606265, 5374062, 3065308, 6918398, 2956279, 8949324, 2804580, 3421479, 7846658, 8895839, 8277589, 1262596, 451779, 9972218, 6378556, 4216958, 7127258, 8593578, 326883, 4737513, 6578257, 7582654, 8675499, 9038961, 7902676, 8874020, 5513073, 631930, 912719, 8394492, 1508363, 455175, 9215635, 6813970, 2021710, 5673212, 184474, 4511247, 4653238, 2218883, 9669544, 295018, 3694660, 1709444, 4019025, 5047809, 45740, 1035395, 8159408, 1557286, 1304144, 6496263, 2094202, 9945315, 1905585, 1143081, 6994125, 9609830, 1077628, 3488222, 6299366, 7187139, 3883908, 7077292, 3210807, 7328762, 7695314, 1138834, 7689433, 5083719, 202831, 8138452, 5495064, 7543763, 1597085, 5429837, 8455839, 6925605, 6600884, 3571512, 3422637, 8911245, 3700762, 2338168, 6830853, 2539094, 490627, 2294717, 497349, 8297867, 7299269, 4769134, 285033, 4335917, 9908413, 152868, 2658658, 3525848, 1884044, 4953877, 8660374, 8989154, 888731, 7217408, 2614940, 7990455, 9779818, 1441488, 9605891, 4518756, 3705442, 9331226, 404585, 9011202, 7355000, 7461968, 6512552, 2689841, 2873446, 256454, 1068037, 8786859, 2323599, 3332506, 2361155, 7476810, 5605915, 5950352, 6491737, 8696129, 4637800, 4207476, 9334774, 840248, 9159149, 5201180, 7211332, 3135016, 8524857, 4566111, 7697488, 1833291, 7227481, 8289951, 2389102, 9102789, 8585135, 1869227, 4082835, 8447466, 4985240, 4176179 };
	cout << "maxProfit=" << maxProfit(nums) << '\n';
	cout << INT_MIN - INT_MAX << '\n';
}
int findCatalanNumRecursive(int n) {
	if (n <= 1)
		return 1;
	int res = 0;
	for (auto i = 0; i < n; ++i) {
		res = res + findCatalanNumRecursive(i) * findCatalanNumRecursive(n - i - 1);
	}
	return res;
}
int findCatalanNumTopDown(int n, vector<int>& memo) {
	if (n <= 1)
		return 1;
	if (memo[n] == 0) {
		for (auto i = 0; i < n; ++i) {
			memo[n] = memo[n] + findCatalanNumTopDown(i,memo) * findCatalanNumTopDown(n - i - 1,memo);
		}
	}
	return memo[n];
}
int uniqueBinarySearchTrees(int A) {
	return findCatalanNumRecursive(A);
}
void uniqueBinarySearchTreesDriver() {
	cout << "uniqueBinarySearchTrees=" << uniqueBinarySearchTrees(3) << '\n';
	vector<int>memo(4);
	cout << "findCatalanNumTopDown=" << findCatalanNumTopDown(3,memo) << '\n';
}
int maxProduct(const vector<int> &A) {
	/*int ans=A[0],j=A[0];
    vector<int> v;
    int i;
    for(i=0;i<A.size();i++){
        if(A[i]!=0)
            v.push_back(A[i]);
        else{
            j=0;
            if(v.size()>0)
                ans=max(ans,B(v));
            v.clear();
        }
    }
    if(v.size()>0)
        ans=max(ans,B(v));
    return max(ans,j);*/
	//DOES not work
	int curr_max = 1;
	int curr_min = 1;
	int max_prod = 1;

	for (auto i = 0; i < A.size(); ++i) {
		if (A[i] > 0) {
			curr_max = curr_max * A[i];
			curr_min = min(curr_min * A[i], 1);
		}
		else if (A[i] == 0) {
			curr_max = 1;
			curr_min = 1;
		}
		else {
			int temp = curr_max;
			curr_max = max(curr_min * A[i], 1);
			curr_min = temp * A[i];
		}
		//update max prod if needed
		max_prod = curr_max > max_prod ? curr_max : max_prod;
	}
	return max_prod;
}
void maxProductDriver() {
	vector<int>nums{ 2, 3, -2, 4 };
	cout << "maxProduct=" << maxProduct(nums) << '\n';
}
int maxSumWithoutAdjacentElements(vector<vector<int>> &A) {
	if (A.empty())
		return 0;
	//choose max element of first column
	int included = max(A[0][0], A[1][0]);
	//exluding element of first column
	int excluded = 0, excluded_new = 0;
	//traverse rest
	for (auto i = 1; i < A[0].size(); ++i) {
		excluded_new = max(included, excluded);
		included = excluded + max(A[0][i], A[1][i]);
		excluded = excluded_new;
	}
	return max(included, excluded);
}
void maxSumWithoutAdjacentElementsDriver() {
	vector<vector<int>>matrix{ {1, 2, 3, 4},{2, 3, 4, 5 } };
	cout << "maxSumWithoutAdjacentElements=" << maxSumWithoutAdjacentElements(matrix) << '\n';
}
int majorityElement(vector<int>& nums) {
	unordered_map<int, int>map;
	for (auto num : nums) {
		++map[num];
	}
	int max_freq = 0;
	int max_val = 0;
	for (auto it = map.begin(); it != map.end(); ++it) {
		if (it->second > max_freq) {
			max_freq = it->second;
			max_val = it->first;
		}
	}
	return max_val;
}
void majorityElementDriver() {
	vector<int>nums{ 2,1,2 };
	cout << "majorityElement=" << majorityElement(nums) << '\n';
}
int gasStation(vector<int>& gas, vector<int>& cost) {
	int sumGas = 0;
	int sumCost = 0;
	int start = 0;
	int tank = 0;

	for (int i = 0; i < gas.size(); i++) {
		sumGas += gas[i];
		sumCost += cost[i];
		tank += gas[i] - cost[i];

		if (tank < 0) {
			start = i + 1;
			tank = 0;
		}
	}
	if (sumGas < sumCost)
		return -1;
	else
		return start;
	/*
	int start = 0;
	int end = 1;
	int curr_gas = gas[start] - cost[start];

	while (start != end || curr_gas < 0) {
		//if we didn't make it, remove route
		while (curr_gas < 0 && start != end) {
			curr_gas = curr_gas - gas[start] - cost[start];
			start = (start + 1) % gas.size();
			//no possible route
			if (start == 0)
				return -1;
		}
		curr_gas = curr_gas + gas[end] - cost[end];
		end = (end + 1) % gas.size();
	}
	return start;*/
}
void gasStationDriver() {
	vector<int>gas{ 1,2 };
	vector<int>cost{ 2,1 };
	cout << "gasStation=" << gasStation(gas, cost) << '\n';
}
int asssignMiceToHoles(vector<int> &mice, vector<int> &holes) {
	if (mice.size() != holes.size())
		return -1;
	//align mice to holes as close as possible
	//by sorting both arrays
	sort(mice.begin(), mice.end());
	sort(holes.begin(), holes.end());
	//find the max difference among mice and holes
	int max_time = 0;
	for (auto i = 0; i < mice.size(); ++i) {
		if (max_time < abs(mice[i] - holes[i]))
			max_time = abs(mice[i] - holes[i]);
	}
	//the max difference represents the longest
	//a mouse will take to get to its hole
	return max_time;
}
void asssignMiceToHolesDriver() {
	vector<int>mice{ 4,-4,2 };
	vector<int>holes{ 4,0,5 };
	cout << "asssignMiceToHoles=" << asssignMiceToHoles(mice, holes) << '\n';
}
int highestProduct(vector<int>& nums) {
	if (nums.size() < 3)
		return -1;
	if (nums.size() == 3)
		return nums[0] * nums[1] * nums[2];
	sort(nums.begin(), nums.end());
	int end = nums.size() - 1;
	return nums[end] * nums[end-1] * nums[end-2];
}
void highestProductDriver() {
	vector<int>nums{ 0,-1,3,100,70,50 };
	cout << "highestProduct=" << highestProduct(nums) << '\n';
}
vector<vector<int>> levelOrder(TreeNode* A) {
	if (A == NULL)
		return { {} };

	vector<vector<int>>level_order;
	queue<TreeNode*>q;
	vector<int>level_values;
	queue<int>level;
	TreeNode* curr = A;
	int curr_level = 0;

	q.push(curr);
	level.push(0);

	while (!q.empty()) {
		curr = q.front();
		curr_level = level.front();
		//moving to the next level
		if (level_order.size() != level.front() + 1)
			level_order.push_back(level_values);

		level_order[curr_level].push_back(curr->val);

		if (curr->left) {
			q.push(curr->left);
			level.push(curr_level + 1);
		}
		if (curr->right) {
			q.push(curr->right);
			level.push(curr_level + 1);
		}
		q.pop();
		level.pop();
	}
	return level_order;
}
void levelOrderDriver() {
	TreeNode* root = new TreeNode(3);
	TreeNode* nine = new TreeNode(9);
	TreeNode* twenty = new TreeNode(20);
	TreeNode* fifteen = new TreeNode(15);
	TreeNode* seven = new TreeNode(7);
	root->left = nine;
	root->right = twenty;
	twenty->left = fifteen;
	twenty->right = seven;
	print2DVector(levelOrder(root));
}
int minSumOfFibonacci(int n) {
	vector<int>fib;
	int count = 0;
	int a = 1, b = 1, temp=0;

	while (a <= n) {
		fib.push_back(a);
		temp = a + b;
		a = b;
		b = temp;
	}
	int num = 0;
	for (auto i = fib.size() - 1; i >= 0; --i) {
		num = fib[i];
		while (num <= n) {
			n = n - num;
			++count;
		}
		if (n == 0)
			break;
	}
	return count;
}
void minSumOfFibonacciDriver() {
	cout << "minSumOfFibonacci=" << minSumOfFibonacci(4) << '\n';
}
struct UndirectedGraphNode {
	int label;
	vector<UndirectedGraphNode *> neighbors;
	UndirectedGraphNode(int x) : label(x) {};
};
UndirectedGraphNode *cloneGraph(UndirectedGraphNode *source) {
	if (source == NULL)
		return NULL;
	unordered_map<UndirectedGraphNode*, UndirectedGraphNode*>map;
	queue<UndirectedGraphNode*>q;
	q.push(source);
	UndirectedGraphNode* node = new UndirectedGraphNode(source->label);
	map[source] = node;
	UndirectedGraphNode* curr = NULL;

	while (!q.empty()) {
		curr = q.front(); q.pop();
		vector<UndirectedGraphNode*> neighbors = curr->neighbors;
		for (auto i = 0; i < neighbors.size(); ++i) {
			//if not cloned yet
			if (map[neighbors[i]] == NULL) {
				node = new UndirectedGraphNode(neighbors[i]->label);
				map[neighbors[i]] = node;
				q.push(neighbors[i]);
			}
			//add neighbors to cloned graph
			map[curr]->neighbors.push_back(map[neighbors[i]]);
		}
	}
	//returned cloned graph
	return map[source];
}
void breadthFirstGraphTraversal(UndirectedGraphNode* source) {
	if (source == NULL)
		return;
	unordered_map<UndirectedGraphNode*, bool>visited;
	queue<UndirectedGraphNode*>q;
	q.push(source);
	visited[source] = true;
	UndirectedGraphNode* curr = NULL;

	while (!q.empty()) {
		curr = q.front(); q.pop();
		cout << curr->label << "->" << curr << '\n';
		vector<UndirectedGraphNode*>n = curr->neighbors;
		for (auto i = 0; i < n.size(); ++i) {
			if (!visited[n[i]]) {
				visited[n[i]] = true;
				q.push(n[i]);
			}
		}
	}
	cout << '\n';
}
void clonedGraphDriver() {
	UndirectedGraphNode *node1 = new UndirectedGraphNode(1);
	UndirectedGraphNode *node2 = new UndirectedGraphNode(2);
	UndirectedGraphNode *node3 = new UndirectedGraphNode(3);
	UndirectedGraphNode *node4 = new UndirectedGraphNode(4);
	vector<UndirectedGraphNode *> v;
	v.push_back(node2);
	v.push_back(node4);
	node1->neighbors = v;
	v.clear();
	v.push_back(node1);
	v.push_back(node3);
	node2->neighbors = v;
	v.clear();
	v.push_back(node2);
	v.push_back(node4);
	node3->neighbors = v;
	v.clear();
	v.push_back(node3);
	v.push_back(node1);
	node4->neighbors = v;
	breadthFirstGraphTraversal(node1);

	UndirectedGraphNode* clone = cloneGraph(node1);
	breadthFirstGraphTraversal(clone);
}
struct Point {
	int x;
	int y;
};
bool isValidIndex(vector<int> &E, vector<int> &F,int r,int x, int y) {
	for (auto i = 0; i < E.size(); ++i) {
		int x1 = E[i];
		int x2 = F[i];
		if ((x == x1) && (y == x2))
			return false;
		int n = (x1 - x) * (x1 - x) + (x2 - y) * (x2 - y);
		if (n <= r)
			return false;
	}
	return true;
}
string validPath(int A, int B, int C, int D, vector<int> &E, vector<int> &F) {
	vector<vector<int>>dp(A + 1, vector<int>(B + 1));
	for (auto i = 0; i <= A; ++i) {
		for (auto j = 0; j <= B; ++j) {
			dp[i][j] = -1;
		}
	}
	dp[0][0] = 1;
	int r = D*D;
	queue<Point>q;
	Point curr;
	curr.x = 0; curr.y = 0;
	q.push(curr);

	vector<int> a1 = { 1,1,1,0,-1,-1,-1,0 };
	vector<int> a2 = { -1,0,1,1,1,0,-1,-1 };

	while (!q.empty()) {
		curr = q.front(); q.pop();
		int x1 = curr.x;
		int x2 = curr.y;

		for (auto i = 0; i < 8; ++i) {
			int t1 = x1 + a1[i];
			int t2 = x2 + a2[i];
			if ((t1 >= 0) && (t1 <= A) && (t2 >= 0) && (t2 <= B)) {
				if (dp[t1][t2] == -1) {
					if (!isValidIndex(E,F,r,t1, t2)) {
						dp[t1][t2] = 2;
					}
					else {
						dp[t1][t2] = 1;
						Point p;
						p.x = t1;
						p.y = t2;
						q.push(p);
					}
				}
			}
		}
		if (dp[A][B] != -1)
			break;
	}
	if (dp[A][B] == 1)
		return "YES";
	return "NO";
}
void validPathDriver() {

}
bool isadjacent(string& a, string& b)
{
	int count = 0;  // to store count of differences
	int n = a.length();

	// Iterate through all characters and return false
	// if there are more than one mismatching characters
	for (int i = 0; i < n; i++)
	{
		if (a[i] != b[i]) count++;
		if (count > 1) return false;
	}
	return count == 1 ? true : false;
}
struct QItem{
	string word;
	int len;
};
int ladderLength(string& start, string& target, set<string> &D){
	//copied from gfg and does not work
	// Create a queue for BFS and insert 'start' as source vertex
	queue<QItem> Q;
	QItem item = { start, 1 };  // Chain length for start word is 1
	Q.push(item);

	// While queue is not empty
	while (!Q.empty())
	{
		// Take the front word
		QItem curr = Q.front();
		Q.pop();

		// Go through all words of dictionary
		for (set<string>::iterator it = D.begin(); it != D.end(); it++)
		{
			// Process a dictionary word if it is adjacent to current
			// word (or vertex) of BFS
			string temp = *it;
			if (isadjacent(curr.word, temp))
			{
				// Add the dictionary word to Q
				item.word = temp;
				item.len = curr.len + 1;
				Q.push(item);

				// Remove from dictionary so that this word is not
				// processed again.  This is like marking visited
				D.erase(temp);

				// If we reached target
				if (temp == target)
					return item.len;
			}
		}
	}
	return 0;
}
void ladderLengthDriver() {
	string start{ "hit" };
	string end{ "cog" };
	set<string>dict;
	dict.insert("hot");
	dict.insert("dot");
	dict.insert("dog");
	dict.insert("lot");
	dict.insert("log");
	cout << "ladderLength=" << ladderLength(start, end, dict) << '\n';
}
vector<string>m;
string v;
bool ok;
int n, c;
int dx[] = { 1,-1,0,0 };
int dy[] = { 0,0,1,-1 };
void solve(int i, int x, int y) {
	if (ok)return;
	if (i == v.size()) {
		ok = true;
		return;
	}
	for (int j = 0; j<4; j++) {
		if (x + dx[j] >= n || x + dx[j]<0 || y + dy[j] >= c || y + dy[j]<0)continue;
		if (m[x + dx[j]][y + dy[j]] == v[i])solve(i + 1, x + dx[j], y + dy[j]);
	}
}
int wordSearchBoard(vector<string> &A, string B) {
	ok = false;
	n = A.size();
	c = A[0].size();
	m = A;
	v = B;
	for (int i = 0; i<A.size(); i++) {
		for (int j = 0; j<A[0].size(); j++) {
			if (A[i][j] == v[0])solve(1, i, j);
		}
	}
	return ok;
}
void wordSearchBoardDriver() {
	vector<string>board{ {"ABCE"},{"SFCS"},{"ADEE"} };
	string word1{ "ABCCED" };
	string word2{ "SEE" };
	string word3{ "ABCB" };
	string word4{ "ABFSAB" };
	string word5{ "ABCD" };
	cout << "wordSearchBoard(1)=" << wordSearchBoard(board, word1) << '\n';
	cout << "wordSearchBoard(1)=" << wordSearchBoard(board, word2) << '\n';
	cout << "wordSearchBoard(1)=" << wordSearchBoard(board, word3) << '\n';
	cout << "wordSearchBoard(1)=" << wordSearchBoard(board, word4) << '\n';
	cout << "wordSearchBoard(0)=" << wordSearchBoard(board, word5) << '\n';
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
	//mergeSortedArraysDriver();
	//arrayThreePointersDriver();
	//reverseListBetweenDriver();
	//reverseLinkedListDriver();
	//removeDuplicatesDriver();
	//insertionSortDriver();
	//insertionSortListDriver();
	//detectCycleDriver();
	//addTwoNumbersDriver();
	//reverseStringStackDriver();
	//validParenthesisDriver();
	//redundantParensDriver();
	//slidingMaximumDriver();
	//minStackDriver();
	//reverseLinkedListRecursiveDriver();
	//combinationsDriver();
	//combinationSumDriver();
	//allPermutationsDriver();
	//solveNQueensDriver();
	//palindromePartitioningDriver();
	//generateParenthesisDriver();
	//twoSumDriver();
	//findSubstringDriver();
	//lengthOfLongestSubstringDriver();
	//solveSudokuDriver();
	//maxPointsDriver();
	//findAnagramsDriver();
	//minWindowDriver();
	//mergeKSortedListsDriver();
	//LeastRecentlyUsedDriver();
	//distinctNumbersDriver();
	//isValidBSTDriver();
	//inOrderDriver();
	//preOrderDriver();
	//postOrderDriver();
	//nextRightPointersDriver();
	//hasPathSumDriver();
	//maxDepthOfBSTDriver();
	//isBalancedBSTDriver();
	//identicalThreesDriver();
	//isSymmetricDriver();
	//leastCommonAncestorDriver();
	//shortestUniquePrefixDriver();
	//flattenBinaryTreeToListDriver();
	//buildTreePreorderInorderDriver();
	//kthsmallestDriver();
	//BSTIteratorDriver();
	//climbStairsDriver();
	//editDistanceDriver();
	//longestIncreasingSubsequenceDriver();
	//canJumpDriver();
	//maxRectangleInMatrixDriver();
	//rodCutDriver();
	//nDigitNumsWithSumDriver();
	//longestValidParenthesesBottomUpDriver();
	//maxProfitDriver();
	//uniqueBinarySearchTreesDriver();
	//maxProductDriver();
	//maxSumWithoutAdjacentElementsDriver();
	//majorityElementDriver();
	//gasStationDriver();
	//asssignMiceToHolesDriver();
	//highestProductDriver();
	//levelOrderDriver();
	//minSumOfFibonacciDriver();
	//clonedGraphDriver();
	//validPathDriver();
	//ladderLengthDriver();
	//wordSearchBoardDriver();

	return 0;
}