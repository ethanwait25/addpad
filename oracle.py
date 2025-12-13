from agent import Agent
from env import Cursor

EMPTY = -1

class Oracle(Agent):
    def act(self, obs):
        a = obs["A"]
        b = obs["B"]
        cursor = Cursor(obs["cursor"])
        pad = obs["pad"]

        def digit(n, place):
            return (n // place) % 10
        
        def carry(pos):
            return pad[pos] if pad[pos] != EMPTY else 0
        
        match cursor:
            case Cursor.ONES_OUT:
                if pad[0] == EMPTY:
                    value = digit(a, 1) + digit(b, 1)
                    return value % 10
                return 11
            
            case Cursor.ONES_CARRY:
                value = digit(a, 1) + digit(b, 1)
                desired = 1 if value >= 10 else 0
                if carry(1) != desired:
                    return 10
                return 11
            
            case Cursor.TENS_OUT:
                if pad[2] == EMPTY:
                    value = digit(a, 10) + digit(b, 10) + carry(1)
                    return value % 10
                return 11
            
            case Cursor.TENS_CARRY:
                value = digit(a, 10) + digit(b, 10) + carry(1)
                desired = 1 if value >= 10 else 0
                if carry(3) != desired:
                    return 10
                return 11
            
            case Cursor.HUNDREDS_OUT:
                if pad[4] == EMPTY:
                    value = digit(a, 100) + digit(b, 100) + carry(3)
                    return value % 10
                return 11
            
            case Cursor.HUNDREDS_CARRY:
                value = digit(a, 100) + digit(b, 100) + carry(3)
                desired = 1 if value >= 10 else 0
                if carry(5) != desired:
                    return 10
                return 11
            
            case Cursor.THOUSANDS_OUT:
                if pad[6] == EMPTY:
                    value = digit(a, 1000) + digit(b, 1000) + carry(5)
                    return value % 10
                return 13
            
        return 11