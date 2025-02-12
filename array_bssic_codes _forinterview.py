# ) Find the Maximum Element in an Array
# arr = [3, 5, 1, 8, 4]
# print(f"Maximum element: {max(arr)}")

# Rotate an Array
# arr = [1, 2, 3, 4, 5]
#  k = 2
# rotated = arr[-k:] + arr[:-k]
# print(f"Rotated array: {rotated}")

# a) Check if a String is a Palindrome
# s = "radaru"
# print(s[::1])
# print(s)
# print(f"Is palindrome: {s == s[::-1]}")

# b) Find the First Non-Repeating Character
# from collections import Counter
# s = "fizzah"
# counter = Counter(s)
# print(counter)
# for char in s:
#     if counter[char] == 1:
#         print(f"First non-repeating character: {char}")
#         break

# Search an Element in a Matrix
# matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# target = 10
# for row in matrix:
#     if target in row:
# # found = any(target in row for row in matrix)
#         print("Element found:",target)

#     else:
#         print("no")


def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    print ("ggggggg:",low,high)
    while low <= high:
        mid = (low + high) // 2
        print(mid)
        print("lllll",arr[mid],target)
        if arr[mid] == target:
            print("%%%%%%%%%%%%%%%%55")
            return mid
        elif arr[mid] < target:
            print('*********8')
            low = mid + 1
        else:
            print("WWWWWWWWWWWWWWWWWw")
            high = mid - 1
    return -1

arr = [1, 3, 5, 7, 9]
print(f"Element found at index: {binary_search(arr, 2)}")
        