def spiral_matrix(matrix):
  """Generates a spiral matrix from the given matrix.

  Args:
    matrix: A 2D list representing the matrix.

  Returns:
    A 2D list representing the spiral matrix.
  """

  result = []
  top, bottom = 0, len(matrix) - 1
  left, right = 0, len(matrix[0]) - 1
  while top <= bottom and left <= right:
    # Add the top row.
    for i in range(left, right + 1):
      result.append(matrix[top][i])
    top += 1

    # Add the right column.
    for i in range(top, bottom + 1):
      result.append(matrix[i][right])
    right -= 1

    # Add the bottom row in reverse order.
    for i in range(right, left - 1, -1):
      result.append(matrix[bottom][i])
    bottom -= 1

    # Add the left column in reverse order.
    for i in range(bottom, top - 1, -1):
      result.append(matrix[i][left])
    left += 1

  return result


# Example usage:
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

print(spiral_matrix(matrix))