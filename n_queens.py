class Solution(object):
    def solveNQueens(self, n):

        def available(row,col):
            # check if it's safe to put a queen here
            if not columns[col] and not diag1[row+col] and not diag2[row-col + n- 1]:
                return True
            else:
                return False
        
        def updateBoard(row, col, flag):
            # flag is False if no queen in the current position
            # flag is a positive integer representing the corresponding row
            if flag:
                columns[col] = row+1
            else:
                columns[col] = flag
            diag1[row+col] = flag
            diag2[row-col + n-1] = flag
        
        
        def makeBoard():
            # we found a solution. Transform the record into a board
            board = ['.'*n ]*n
            for col, row in enumerate(columns):
                board[row-1] = '.'*(col) + 'Q' + '.'*(n-col-1)
            res.append(board)
            
        def findNextRow(row):
            # if we have filled n rows
            if row == n:
                makeBoard()
                return
            
            # try every column
            for col in range(n):
                if not available(row, col): continue
                updateBoard(row, col, True)
                findNextRow(row+1) 
                updateBoard(row, col, False) # backtrack
                
        # record the board
        columns = [0]*n
        diag1, diag2 = [0]*(2*n-1), [0]*(2*n-1)
        res = []    
        findNextRow(0)
        return res

ty=Solution()
num=4
ans=ty.solveNQueens(num)
print("%d solutions are possible for N = %d"%(len(ans),num))
print(" ")
for i in range(len(ans)):
    print("solution %d"%(i+1))
    for j in ans[i]:
        print(j)
    print("  ")