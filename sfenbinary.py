##Melanie Fernandez
##Neural Networks Program: Binary Strings

import sqlite3

def getsql():
    hands_value = []
    conn = sqlite3.connect("shogi_board_states.db")
    cursor = conn.cursor()
    cursor.execute(f'SELECT board || " " || hands || " " || winner FROM board_states')
    all_rows = [row[0] for row in cursor.fetchall()]
    return all_rows

    cursor.close()


def convert_int_to_zeroes(number):
    number = int(number)
    number *= 5
    text = '0'*number
    return text


def convert_hand(hand):
    result = ''
    i = 0
    while i < len(hand):
        if hand[i].isdigit():
            count = int(hand[i])
            #moves to the next character
            i +=1
            piece = hand[i]
            result += piece * count
        else:
            result += hand[i]
        
        i += 1

    return result
    binary_hand = convert_int_to_zeroes(result)
    return binary_hand



def turn_into_binary(sfen):
    sfen_mapping = {
        'P': '11111', 'L': '00001', 'N': '00010', 'S': '00011',
        'G': '00100', 'K': '00101', 'R': '00110', 'B': '00111',
        '+P': '01000', '+L': '01001', '+N': '01010', '+S': '01011',
        '+R': '01100', '+B': '01101', '+G': '01110', 
        'p': '10000', 'l': '10001', 'n': '10010', 's': '10011',
        'g': '10100', 'k': '10101', 'r': '10110', 'b': '10111',
        '+p': '11000', '+l': '11001', '+n': '11010', '+s': '11011',
        '+r': '11100', '+b': '11101', '+g': '11110', 
        'z': '1' , 'w': "0"

        }

    binary_list = ''

    for char in sfen:
        if char in sfen_mapping:
            binary_list += (sfen_mapping[char])
        else:
            binary_list += (char)

    return binary_list



def main():
    allSfens = getsql()
    convertedSfens = ''


    for i in range(len(allSfens)):
        sfen = allSfens[i].split(' ')

        if len(sfen) < 2:
            continue 

        board = sfen[0]
        hand = sfen[1]

        for character in board: 
            try:
                board = board.replace(character, convert_int_to_zeroes(character))
            except:
                pass

        if board.endswith('b'):
            board = turn_into_binary(board[:-1]+'z')
        else:
            board = turn_into_binary(board)

        hand = convert_hand(hand)
        sfen_combined = f'{board} {hand}'

        convertedSfens.join(sfen_combined)

       result = []
       for i in range (0 , len(convertedSfens), 45):
           chunk = convertedSfens [i:i+45]
           result.append(chunk)

        result = '\n'.join(result)
    return result


        #allSfens[i] = (" ".join(sfen))
    #return allSfens

        #print(" ".join(sfen))

result = main()
if result:
    print(result[0])
else:
    print("No valid data found")
    




 
