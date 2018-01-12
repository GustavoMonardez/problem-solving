#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <stack>
#include <deque>

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
	minStackDriver();

	return 0;
}