from matplotlib.pyplot import yscale
import numpy as np

class CandyCrushGym:

    def __init__(self):

        self.FIELD_SIZE = 8
        self.NUM_DIRECTIONS = 4
        self.NUM_ELEMENTS = 7
        self.REWARD_PER_HIT = 0.25

        self.COLOR_BOMB_CANDY_ID = 25

        self.columns_to_fill = set()

        self.reset()

    def reset(self):

        rnds = np.random.randint(1, self.NUM_ELEMENTS, size=self.FIELD_SIZE*self.FIELD_SIZE, dtype=np.int8)
        self.state = np.reshape(rnds, newshape=(self.FIELD_SIZE,self.FIELD_SIZE))

        #self.state = np.array([[4,4,4,4,4],[1,2,12,0,4],[4,5,-1,-1,-1], [4,0,0,0,4], [0,0,0,0,5]])
    
        return self.state

    def isValidAction(self, action):

        fieldID = action // self.NUM_DIRECTIONS

        direction = action % self.NUM_DIRECTIONS

        x = fieldID // self.FIELD_SIZE
        y = fieldID % self.FIELD_SIZE

        # Swap candy
        x_swap = x # attention: numpy x->y are swapped
        y_swap = y # attention: numpy x->y are swapped
        # top
        if direction == 0:
            y_swap += -1
        # down
        elif direction == 1: 
            y_swap += 1
        # right 
        elif direction == 2:
            x_swap += 1
        # left 
        elif direction == 3:
            x_swap += -1

        print(f"{action}: ({x}, {y}) ({x_swap}, {y_swap}) -> {self.isValidIndex(x,y) and self.isValidIndex(x_swap, y_swap)}")
        return self.isValidIndex(x,y) and self.isValidIndex(x_swap, y_swap)
           

    def step(self, action):
        
        if not self.isValidAction(action):
            raise ValueError("Invalid action")

        fieldID = action // self.NUM_DIRECTIONS

        direction = action % self.NUM_DIRECTIONS

        x = fieldID // (self.FIELD_SIZE * self.FIELD_SIZE)
        y = fieldID % (self.FIELD_SIZE * self.FIELD_SIZE)

        # Swap candy
        x_swap = x # attention: numpy x->y are swapped
        y_swap = y # attention: numpy x->y are swapped
        # top
        if direction == 0:
            y_swap += -1
        # down
        elif direction == 1: 
            y_swap += 1
        # right 
        elif direction == 2:
            x_swap += 1
        # left 
        elif direction == 3:
            x_swap += -1
        

        # swap
        tmp = self.state[y,x]
        self.state[y,x] = self.state[y_swap, x_swap]
        self.state[y_swap, x_swap] = tmp

        reward = self.react(x,y, x_swap, y_swap)

        if reward == 0:

            # swap again -> undo previous swap
            tmp = self.state[y,x]
            self.state[y,x] = self.state[y_swap, x_swap]
            self.state[y_swap, x_swap] = tmp
        else:
            # Shift column to remove -1 (hit elements)
            for column_idx in self.columns_to_fill:
                
                for _ in range(self.FIELD_SIZE):
                    for i in range(self.FIELD_SIZE):
                        
                        if self.state[i,column_idx] == -1:
                            if i - 1 < 0:
                                x = np.random.randint(1, self.NUM_ELEMENTS)
                            else:
                                x = self.state[i - 1,column_idx]
                                self.state[i - 1,column_idx] = -1
                            self.state[i,column_idx] = x
        
        self.columns_to_fill = set()
     
        return self.state, reward

    
    def step_display(self, action):
        
        if not self.isValidAction(action):
            raise ValueError("Invalid action")

        fieldID = action // self.NUM_DIRECTIONS

        direction = action % self.NUM_DIRECTIONS

        x = fieldID // self.FIELD_SIZE
        y = fieldID % self.FIELD_SIZE

        # Swap candy
        x_swap = x # attention: numpy x->y are swapped
        y_swap = y # attention: numpy x->y are swapped
        # top
        if direction == 0:
            y_swap += -1
        # down
        elif direction == 1: 
            y_swap += 1
        # right 
        elif direction == 2:
            x_swap += 1
        # left 
        elif direction == 3:
            x_swap += -1
        

        # swap
        tmp = self.state[y,x]
        self.state[y,x] = self.state[y_swap, x_swap]
        self.state[y_swap, x_swap] = tmp

        reward = self.react(x,y, x_swap, y_swap)

        if reward == 0:

            # swap again -> undo previous swap
            tmp = self.state[y,x]
            self.state[y,x] = self.state[y_swap, x_swap]
            self.state[y_swap, x_swap] = tmp
        
        columns_to_fill = self.columns_to_fill
        
        self.columns_to_fill = set()
     
        return self.state, reward, columns_to_fill
    
    def react(self,x,y, x_swap, y_swap):
        
        reward = 0

        candy_id = self.state[y,x]
        swapped_candyID = self.state[y_swap,x_swap]

        #
        # Color bomb candy
        #

        reward += self.reactToColorBomb(x_swap, y_swap,swapped_candyID, x,y)
        reward += self.reactToColorBomb(x,y,candy_id, x_swap, y_swap)
           
        #
        # Wrapped + Stripped candy
        #

        isMatch = False
        if self.isWrappedCandyID(candy_id):
            id1 = self.convertWrappedCandy_toNormal(candy_id)
            
            if self.isHorizontalStrippedCandy(swapped_candyID):
                id2 = self.convertHorizontalStrippedCandy_toNormal(swapped_candyID)
                if id1 == id2:
                    isMatch = True
            
            if self.isVerticalStrippedCandy(swapped_candyID):
                id2 = self.convertVerticalStrippedCandy_toNormal(swapped_candyID)
                if id1 == id2:
                    isMatch = True
        
        if self.isHorizontalStrippedCandy(candy_id):
            if self.isWrappedCandyID(swapped_candyID):
                id1 = self.convertHorizontalStrippedCandy_toNormal(candy_id)
                id2 = self.convertWrappedCandy_toNormal(swapped_candyID)
                if id1 == id2:
                    isMatch = True
        
        if self.isVerticalStrippedCandy(candy_id):

            if self.isWrappedCandyID(swapped_candyID):
                id1 = self.convertVerticalStrippedCandy_toNormal(candy_id)
                id2 = self.convertWrappedCandy_toNormal(swapped_candyID)
                if id1 == id2:
                    isMatch = True
   
        if isMatch:
            self.state[max(y_swap-1, 0):min(y_swap+2, self.FIELD_SIZE -1), :] = -1
            self.state[:, max(x_swap-1,0):min(x_swap+2,self.FIELD_SIZE - 1)] = -1

            for i in range(0, self.FIELD_SIZE):
                self.columns_to_fill.add(i)

            reward += 1.5

        list_rows, list_columns = self.scanRowsColums(x,y, candy_id)
        list_rows_swap, list_columns_swap = self.scanRowsColums(x_swap,y_swap, swapped_candyID)

        #
        # Five candy in a row/column -> create color bomb candy
        #

        reward += self.reactFiveCandys(x_swap, y_swap, list_rows_swap, list_columns_swap)
        reward += self.reactFiveCandys(x_swap, y_swap, list_rows, list_columns)

        #
        # Wrapped candy + candy (same color)
        #

        reward += self.reactWrappedCandy(swapped_candyID, x_swap, y_swap, candy_id)
        reward += self.reactWrappedCandy(candy_id, x_swap, y_swap, swapped_candyID)
        

        #
        # Horizontal or vertical
        #

        reward += self.react_horizontal_vertical(list_rows_swap, list_columns_swap)
        reward += self.react_horizontal_vertical(list_rows, list_columns)
        
        #
        # Five candys: T or L shape
        #

        reward_l_t_1 = self.react_t_l_shape(swapped_candyID, x_swap, y_swap, list_rows_swap, list_columns_swap)
        reward_l_t_2 = self.react_t_l_shape(candy_id, x_swap, y_swap, list_rows, list_columns)

        if reward_l_t_1 + reward_l_t_2 != 0:
            return reward + reward_l_t_1 + reward_l_t_2

        #
        # Four candys in a line (column or row) 
        #
        
        reward += self.reactFourCandys(swapped_candyID, x_swap, y_swap, list_rows_swap, list_columns_swap)
        reward += self.reactFourCandys(candy_id, x_swap, y_swap, list_rows, list_columns)
        
        #
        # Three candys in a line (column or row)
        #

        reward += self.reactThreeCandys(list_rows_swap, list_columns_swap)
        reward += self.reactThreeCandys(list_rows, list_columns)
        
        return reward

    def is_T_shape(self, candy_id, list_rows, list_columns):

   
        if len(list_rows) == 3:

            # 1 1 1 
            # 0 1 0
            # 0 1 0
            y0, x0 = list_rows[1]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0+2,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0+2,x0]:
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0+2,x0))
                    return True, list_rows
            
            # 1 0 0 
            # 1 1 1
            # 1 0 0
            y0, x0 = list_rows[0]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0-1,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0-1,x0]:
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0-1,x0))
                    return True, list_rows

            # 0 1 0 
            # 0 1 0
            # 1 1 1
            y0, x0 = list_rows[1]
            if self.isValidIndex(y0-1,x0) and self.isValidIndex(y0-2,x0):
                if candy_id == self.state[y0-1,x0] == self.state[y0-2,x0]:
                    list_rows.append((y0-1,x0))
                    list_rows.append((y0-2,x0))
                    return True, list_rows
            
            # 0 0 1 
            # 1 1 1
            # 0 0 1
            y0, x0 = list_rows[2]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0-1,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0-1,x0]:
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0-1,x0))
                    return True, list_rows
        
        elif len(list_rows) == 4:

            # 1 1 1 1
            # 0 1 0 0
            # 0 1 0 0
            y0, x0 = list_rows[1]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0+2,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0+2,x0]:
                    list_rows = list_rows[:-1]
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0+2,x0))
                    return True, list_rows
            
            # 1 0 0 0
            # 1 1 1 1
            # 1 0 0 0
            y0, x0 = list_rows[0]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0-1,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0-1,x0]:
                    list_rows = list_rows[:-1]
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0-1,x0))
                    return True, list_rows
            
            # 1 1 1 1
            # 0 0 1 0
            # 0 0 1 0
            y0, x0 = list_rows[2]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0+2,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0+2,x0]:
                    list_rows = list_rows[1:]
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0+2,x0))
                    return True, list_rows
            
            # 0 0 0 1
            # 1 1 1 1
            # 0 0 0 1
            y0, x0 = list_rows[3]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0-1,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0-1,x0]:
                    list_rows = list_rows[1:]
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0-1,x0))
                    return True, list_rows
            
            # 0 1 0 0
            # 0 1 0 0
            # 1 1 1 1
            y0, x0 = list_rows[1]
            if self.isValidIndex(y0-1,x0) and self.isValidIndex(y0-2,x0):
                if candy_id == self.state[y0-1,x0] == self.state[y0-2,x0]:
                    list_rows = list_rows[:-1]
                    list_rows.append((y0-1,x0))
                    list_rows.append((y0-2,x0))
                    return True, list_rows

            # 0 0 1 0
            # 0 0 1 0
            # 1 1 1 1
            y0, x0 = list_rows[2]
            if self.isValidIndex(y0-1,x0) and self.isValidIndex(y0-2,x0):
                if candy_id == self.state[y0-1,x0] == self.state[y0-2,x0]:
                    list_rows = list_rows[1:]
                    list_rows.append((y0-1,x0))
                    list_rows.append((y0-2,x0))
                    return True, list_rows


        elif len(list_columns) == 3:

            # 1 1 1 
            # 0 1 0 
            # 0 1 0 
            y0, x0 = list_columns[0]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0+1):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0+1]:
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0+1))
                    return True, list_columns

            # 1 0 0 
            # 1 1 1 
            # 1 0 0 
            y0, x0 = list_columns[1]
            if self.isValidIndex(y0,x0+1) and self.isValidIndex(y0,x0+2):
                if candy_id == self.state[y0,x0+1] == self.state[y0,x0+2]:
                    list_columns.append((y0,x0+1))
                    list_columns.append((y0,x0+2))
                    return True, list_columns
            
            # 0 1 0 
            # 0 1 0 
            # 1 1 1
            y0, x0 = list_columns[2]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0+1):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0+1]:
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0+1))
                    return True, list_columns
            
            # 0 0 1 
            # 1 1 1 
            # 0 0 1
            y0, x0 = list_columns[1]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0-2):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0-2]:
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0-2))
                    return True, list_columns
        
        elif len(list_columns) == 4:

            # 1 1 1 0 
            # 0 1 0 0
            # 0 1 0 0
            # 0 1 0 0
            y0, x0 = list_columns[0]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0+1):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0+1]:
                    list_columns = list_columns[:-1]
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0+1))
                    return True, list_columns
            
            # 0 1 0 0 
            # 0 1 1 1
            # 0 1 0 0
            # 0 1 0 0
            y0, x0 = list_columns[1]
            if self.isValidIndex(y0,x0+1) and self.isValidIndex(y0,x0+2):
                if candy_id == self.state[y0,x0+1] == self.state[y0,x0+2]:
                    list_columns = list_columns[:-1]
                    list_columns.append((y0,x0+1))
                    list_columns.append((y0,x0+2))
                    return True, list_columns
            
            # 0 1 0 0 
            # 0 1 0 0
            # 0 1 1 1
            # 0 1 0 0
            y0, x0 = list_columns[2]
            if self.isValidIndex(y0,x0+1) and self.isValidIndex(y0,x0+2):
                if candy_id == self.state[y0,x0+1] == self.state[y0,x0+2]:
                    list_columns = list_columns[1:]
                    list_columns.append((y0,x0+1))
                    list_columns.append((y0,x0+2))
                    return True, list_columns
            
            # 0 1 0 0 
            # 0 1 0 0
            # 0 1 0 0
            # 1 1 1 0
            y0, x0 = list_columns[3]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0+1):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0+1]:
                    list_columns = list_columns[1:]
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0+1))
                    return True, list_columns
            
            # 0 0 1 0 
            # 1 1 1 0
            # 0 0 1 0
            # 0 0 1 0
            y0, x0 = list_columns[1]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0-2):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0-2]:
                    list_columns = list_columns[:-1]
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0-2))
                    return True, list_columns
            
            # 0 0 1 0 
            # 0 0 1 0
            # 1 1 1 0
            # 0 0 1 0
            y0, x0 = list_columns[2]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0-2):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0-2]:
                    list_columns = list_columns[1:]
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0-2))
                    return True, list_columns
        
        return False, []

    def is_L_shape(self, candy_id, list_rows, list_columns):
   
        if len(list_rows) == 3:

            # 1 0 0
            # 1 0 0
            # 1 1 1
            y0, x0 = list_rows[0]
            if self.isValidIndex(y0-1,x0) and self.isValidIndex(y0-2,x0):
                if candy_id == self.state[y0-1,x0] == self.state[y0-2,x0]:
                    list_rows.append((y0-1,x0))
                    list_rows.append((y0-2,x0))
                    return True, list_rows
            
            # 1 1 1
            # 1 0 0
            # 1 0 0
            y0, x0 = list_rows[0]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0+2,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0+2,x0]:
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0+2,x0))
                    return True, list_rows
            
            # 1 1 1
            # 0 0 1
            # 0 0 1
            y0, x0 = list_rows[-1]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0+2,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0+2,x0]:
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0+2,x0))
                    return True, list_rows
            
            # 0 0 1
            # 0 0 1
            # 1 1 1
            y0, x0 = list_rows[2]
            if self.isValidIndex(y0-1,x0) and self.isValidIndex(y0-2,x0):
                if candy_id == self.state[y0-1,x0] == self.state[y0-2,x0]:
                    list_rows.append((y0-1,x0))
                    list_rows.append((y0-2,x0))
                    return True, list_rows

        elif len(list_rows) == 4:

            # 1 0 0 0
            # 1 0 0 0
            # 1 1 1 1
            y0, x0 = list_rows[0]
            if self.isValidIndex(y0-1,x0) and self.isValidIndex(y0-2,x0):
                if candy_id == self.state[y0-1,x0] == self.state[y0-2,x0]:
                    list_rows = list_rows[:-1]
                    list_rows.append((y0-1,x0))
                    list_rows.append((y0-2,x0))
                    return True, list_rows
            
            # 1 1 1 1
            # 1 0 0 0
            # 1 0 0 0
            y0, x0 = list_rows[0]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0+2,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0+2,x0]:
                    list_rows = list_rows[:-1]
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0+2,x0))
                    return True, list_rows
            
            # 1 1 1 1
            # 0 0 0 1
            # 0 0 0 1
            y0, x0 = list_rows[-1]
            if self.isValidIndex(y0+1,x0) and self.isValidIndex(y0+2,x0):
                if candy_id == self.state[y0+1,x0] == self.state[y0+2,x0]:
                    list_rows = list_rows[1:]
                    list_rows.append((y0+1,x0))
                    list_rows.append((y0+2,x0))
                    return True, list_rows
            
            # 0 0 0 1
            # 0 0 0 1
            # 1 1 1 1
            y0, x0 = list_rows[-1]
            if self.isValidIndex(y0-1,x0) and self.isValidIndex(y0-2,x0):
                if candy_id == self.state[y0-1,x0] == self.state[y0-2,x0]:
                    list_rows = list_rows[1:]
                    list_rows.append((y0-1,x0))
                    list_rows.append((y0-2,x0))
                    return True, list_rows
            

        elif 3 == len(list_columns):
            
            # 1 0 0
            # 1 0 0
            # 1 1 1
            y0, x0 = list_columns[-1]
            if self.isValidIndex(y0,x0+1) and self.isValidIndex(y0,x0+2):
                if candy_id == self.state[y0,x0+1] == self.state[y0,x0+2]:
                    list_columns.append((y0,x0+1))
                    list_columns.append((y0,x0+2))
                    return True, list_columns
            
            # 1 1 1
            # 1 0 0
            # 1 0 0
            y0, x0 = list_columns[0]
            if self.isValidIndex(y0,x0+1) and self.isValidIndex(y0,x0+2):
                if candy_id == self.state[y0,x0+1] == self.state[y0,x0+2]:
                    list_columns.append((y0,x0+1))
                    list_columns.append((y0,x0+2))
                    return True, list_columns

            # 1 1 1 
            # 0 1 0
            # 0 1 0
            y0, x0 = list_columns[0]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0-2):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0-2]:
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0-2))
                    return True, list_columns

            # 0 0 1
            # 0 0 1
            # 1 1 1
            y0, x0 = list_columns[-1]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0-2):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0-2]:
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0-2))
                    return True, list_columns
        
        elif 4 == len(list_columns):
            
            # 1 0 0
            # 1 0 0
            # 1 0 0
            # 1 1 1
            y0, x0 = list_columns[-1]
            if self.isValidIndex(y0,x0+1) and self.isValidIndex(y0,x0+2):
                if candy_id == self.state[y0,x0+1] == self.state[y0,x0+2]:
                    list_columns = list_columns[1:]
                    list_columns.append((y0,x0+1))
                    list_columns.append((y0,x0+2))
                    return True, list_columns

            # 1 1 1
            # 1 0 0
            # 1 0 0
            # 1 0 0
            y0, x0 = list_columns[0]
            if self.isValidIndex(y0,x0+1) and self.isValidIndex(y0,x0+2):
                if candy_id == self.state[y0,x0+1] == self.state[y0,x0+2]:
                    list_columns = list_columns[:-1]
                    list_columns.append((y0,x0+1))
                    list_columns.append((y0,x0+2))
                    return True, list_columns
            
            # 1 1 1
            # 0 0 1
            # 0 0 1
            # 0 0 1
            y0, x0 = list_columns[0]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0-2):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0-2]:
                    list_columns = list_columns[:-1]
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0-2))
                    return True, list_columns
            
            # 0 0 1
            # 0 0 1
            # 0 0 1
            # 1 1 1
            y0, x0 = list_columns[-1]
            if self.isValidIndex(y0,x0-1) and self.isValidIndex(y0,x0-2):
                if candy_id == self.state[y0,x0-1] == self.state[y0,x0-2]:
                    list_columns = list_columns[1:]
                    list_columns.append((y0,x0-1))
                    list_columns.append((y0,x0-2))
                    return True, list_columns

        return False, []

    def isValidIndex(self, x, y):
        if self.FIELD_SIZE <= x or self.FIELD_SIZE <= y or x < 0 or y < 0:
            return False
        
        return True

    
    def getWrappedCandyID(self, candyID):
        return candyID + 6

    def getHorizontalStrippedCandyID(self, candyID):
        return candyID + 12
    
    def getVerticalStrippedCandyID(self, candyID):
        return candyID + 18

    
    def isNormalCandy(self, candyID):
        return 1 <= candyID <= 6
    
    def isWrappedCandyID(self, candyID):
        return 7 <= candyID <= 12
    
    def isHorizontalStrippedCandy(self, candyID):
        return 13 <= candyID <= 18
 
    def isVerticalStrippedCandy(self, candyID):
        return 19 <= candyID <= 24

    def convertWrappedCandy_toNormal(self, candyID):
        return candyID - 6
    
    def convertHorizontalStrippedCandy_toNormal(self, candyID):
        return candyID - 12
    
    def convertVerticalStrippedCandy_toNormal(self, candyID):
        return candyID - 18

    def equalCandys(self, candyID_1, candyID_2):
       
        if self.isNormalCandy(candyID_1):
            id1 = candyID_1
        elif self.isWrappedCandyID(candyID_1):
            id1 = self.convertWrappedCandy_toNormal(candyID_1)
        elif self.isHorizontalStrippedCandy(candyID_1):
            id1 = self.convertHorizontalStrippedCandy_toNormal(candyID_1)
        elif self.isVerticalStrippedCandy(candyID_1):
            id1 = self.convertVerticalStrippedCandy_toNormal(candyID_1)
        else:
            return False

        if self.isNormalCandy(candyID_2):
            id2 = candyID_2
        elif self.isWrappedCandyID(candyID_2):
            id2 = self.convertWrappedCandy_toNormal(candyID_2)
        elif self.isHorizontalStrippedCandy(candyID_2):
            id2 = self.convertHorizontalStrippedCandy_toNormal(candyID_2)
        elif self.isVerticalStrippedCandy(candyID_2):
            id2 = self.convertVerticalStrippedCandy_toNormal(candyID_2)
        else:
            return False

        return id1 == id2
    

    def reactToColorBomb(self, x,y, candy_id, x_swap, y_swap):
        if candy_id == self.COLOR_BOMB_CANDY_ID:
            self.state[y,x] = -1 # remove candy which "hit" the color bomb
            self.columns_to_fill.add(x)
            kill_allCandys_ID = self.state[y_swap,x_swap]

            cnt = 0
            for x0 in range(self.FIELD_SIZE):
                for y0 in range(self.FIELD_SIZE):

                    if self.state[y0, x0] == kill_allCandys_ID:
                        self.state[y0, x0] = -1
                        self.columns_to_fill.add(x0)
                        cnt += 0.25
            return min(2.0, cnt)
        
        return 0


    def scanRowsColums(self, x,y, candy_id):
        list_rows = []
        list_columns = []

        # count number of candys in a row
        for i in range(0, self.FIELD_SIZE - x):

            if self.equalCandys(self.state[y, x+i], candy_id):
                list_rows.append((y,x+i))
            else:
                break

        for i in reversed(range(0, x)):
        
            if self.equalCandys(self.state[y, i], candy_id):
                list_rows.append((y,i))
            else:
                break
        

        # count number of candys in a column
        for i in range(0, self.FIELD_SIZE - y):
            
            if self.equalCandys(self.state[y + i, x], candy_id):
                list_columns.append((y+i,x))
            else:
                break

        for i in reversed(range(0, y)):
            
            if self.equalCandys(self.state[i, x], candy_id):
                list_columns.append((i,x))
            else:
                break
        
        return list_rows, list_columns

    def reactFiveCandys(self, x, y, list_rows, list_columns):

        state = self.state 

        if len(list_columns) == 5:
            for y0,x0 in list_columns:

                if self.state[y0, x0] == -1:
                    self.state = state
                    return 0

                self.state[y0, x0] = -1
                self.columns_to_fill.add(x0)
            self.state[y, x] = self.COLOR_BOMB_CANDY_ID
            return 1.5
 
        if len(list_rows) == 5:
            for y0,x0 in list_rows:

                if self.state[y0, x0] == -1:
                    self.state = state
                    return 0

                self.state[y0, x0] = -1
                self.columns_to_fill.add(x0)
            self.state[y, x] = self.COLOR_BOMB_CANDY_ID
            return 1.5
        
        return 0
    
    def reactWrappedCandy(self,candy_id, x,y, swapped_candyID):
     
        if self.isWrappedCandyID(candy_id):
            id = self.convertWrappedCandy_toNormal(candy_id)

            if id == swapped_candyID:
                self.state[max(y-1, 0):min(y+2, self.FIELD_SIZE -1), max(x-1, 0):min(x+2, self.FIELD_SIZE -1)] = -1
                
                for i in range(max(x-1, 0),min(x+2, self.FIELD_SIZE)):
                    self.columns_to_fill.add(i)

                return 1
        
        return 0
    
    def react_horizontal_vertical(self, list_rows, list_columns):
        if 2 <= len(list_columns):
            for y0,x0 in list_columns:
                if self.isVerticalStrippedCandy(self.state[y0, x0]):
                    self.state[:, x0] = -1
                    self.columns_to_fill.add(x0)
                    return 1
                
                if self.isHorizontalStrippedCandy(self.state[y0, x0]):
                    self.state[y0, :] = -1

                    for i in range(0, self.FIELD_SIZE):
                        self.columns_to_fill.add(i)

                    return 1
        
        if 2 <= len(list_rows):
            for y0,x0 in list_rows:
                if self.isVerticalStrippedCandy(self.state[y0, x0]):
                    self.state[:, x0] = -1
                    self.columns_to_fill.add(x0)
                    return 1
                
                if self.isHorizontalStrippedCandy(self.state[y0, x0]):
                    self.state[y0, :] = -1
                    for i in range(0, self.FIELD_SIZE):
                        self.columns_to_fill.add(i)

                    return 1
        
        return 0
    
    def react_t_l_shape(self, candy_id, x,y, list_rows, list_columns):

        state = self.state

        list_rows.sort(key=lambda tup: tup[1])
        list_columns.sort(key=lambda tup: tup[0])

        is_T_shape, list_to_remove = self.is_T_shape(candy_id, list_rows, list_columns)

        if is_T_shape:
            
            for y0, x0 in list_to_remove:
                if self.state[y0, x0] == -1:
                    self.state = state
                    return 0
                self.state[y0, x0] = -1
                self.columns_to_fill.add(x0)

            self.state[y, x] = self.getWrappedCandyID(candy_id)
            
            return 0.75
        else:
            is_L_shape, list_to_remove = self.is_L_shape(candy_id, list_rows, list_columns)

            if is_L_shape:
                
                for y0, x0 in list_to_remove:
                    if self.state[y0, x0] == -1:
                        self.state = state
                        return 0
                    self.state[y0, x0] = -1
                    self.columns_to_fill.add(x0)
                    
                self.state[y, x] = self.getWrappedCandyID(candy_id)
                
                return 0.75
        
        return 0
    
    def reactFourCandys(self, candy_id, x, y, list_rows, list_columns):
        state = self.state

        if len(list_columns) == 4:
            for y0, x0 in list_columns:
                if self.state[y0, x0] == -1:
                    self.state = state
                    return 0 
                self.state[y0, x0] = -1
                self.columns_to_fill.add(x0)

            if self.isWrappedCandyID(candy_id):
                candy_id = self.convertWrappedCandy_toNormal(candy_id)

            self.state[y, x] = self.getVerticalStrippedCandyID(candy_id)
            return 0.5
            
        elif len(list_rows) == 4:
            for y0, x0 in list_rows:
                if self.state[y0, x0] == -1:
                    self.state = state
                    return 0 

                self.state[y0, x0] = -1
                self.columns_to_fill.add(x0)

            if self.isWrappedCandyID(candy_id):
                candy_id = self.convertWrappedCandy_toNormal(candy_id)
            self.state[y, x] = self.getHorizontalStrippedCandyID(candy_id)
         
            return 0.5
        
        return 0
    
    def reactThreeCandys(self, list_rows, list_columns):
        state = self.state

        if len(list_columns) == 3:
            for y0, x0 in list_columns:

                if self.state[y0, x0] == -1:
                    self.state = state
                    return 0 

                self.state[y0, x0] = -1   
                self.columns_to_fill.add(x0) 
            return 0.25

        elif len(list_rows) == 3:
            for y0, x0 in list_rows:
                
                if self.state[y0, x0] == -1:
                    self.state = state
                    return 0 

                self.state[y0, x0] = -1
                self.columns_to_fill.add(x0)

            return 0.25
        
        return 0

    @property
    def observation_space_shape(self):
        return (self.FIELD_SIZE, self.FIELD_SIZE)
    
    @property
    def action_space_n(self):
        return self.FIELD_SIZE * self.FIELD_SIZE * self.NUM_DIRECTIONS