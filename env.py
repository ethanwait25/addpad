from dataclasses import dataclass, field
from enum import IntEnum
from random import randint

# 0-9: set digit at cursor
# 10: toggle carry at cursor
# 11: move cursor left
# 12: move cursor right
# 13: be done

EMPTY = -1

class Cursor(IntEnum):
    ONES_OUT = 1
    ONES_CARRY = 2
    TENS_OUT = 3
    TENS_CARRY = 4
    HUNDREDS_OUT = 5
    HUNDREDS_CARRY = 6
    THOUSANDS_OUT = 7

@dataclass
class Scratchpad:
    ones: int = EMPTY
    ones_carry: int = EMPTY
    tens: int = EMPTY
    tens_carry: int = EMPTY
    hundreds: int = EMPTY
    hundreds_carry: int = EMPTY
    thousands: int = EMPTY

@dataclass
class EnvState:
    A: int
    B: int
    target: int
    cursor: Cursor = Cursor.ONES_OUT
    pad: Scratchpad = field(default_factory=Scratchpad)
    steps: int = 0
    done: bool = False

class AddPad:
    def __init__(self, max_digits=3):
        self.max_digits = max_digits
        self.max_steps = 50
        self.initialize_state()

    def step(self, action: int):
        state = self.state
        info = { "illegal": False }
        reward: float = 0.0

        state.steps += 1

        if state.done:
            return self.get_obs(), reward, True, info
        
        if 0 <= action <= 9:
            # 0-9: set digit at cursor
            if state.cursor in {
                Cursor.ONES_OUT,
                Cursor.TENS_OUT,
                Cursor.HUNDREDS_OUT,
                Cursor.THOUSANDS_OUT
            }:
                self.set_digit(action)
            else:
                info["illegal"] = True
        elif action == 10:
            # 10: toggle carry at cursor
            if state.cursor in {
                Cursor.ONES_CARRY,
                Cursor.TENS_CARRY,
                Cursor.HUNDREDS_CARRY
            }:
                self.toggle_carry()
            else:
                info["illegal"] = True
        elif action == 11:
            # 11: move cursor left
            if state.cursor != Cursor.THOUSANDS_OUT:
                self.move_cursor_left()
            else:
                info["illegal"] = True
        elif action == 12:
            # 12: move cursor right
            if state.cursor != Cursor.ONES_OUT:
                self.move_cursor_right()
            else:
                info["illegal"] = True
        elif action == 13:
            # 13: be done
            self.be_done()
        else:
            info["illegal"] = True

        if info["illegal"]:
            reward += -0.1

        reward += -0.01
        
        if not state.done and state.steps >= self.max_steps:
            state.done = True
            info["max_steps"] = True

        if state.done:
            reward += 1.0 if self.is_correct() else -1.0

        return self.get_obs(), reward, state.done, info

    def get_obs(self):
        s = self.state
        return {
            "A": s.A,
            "B": s.B,
            "cursor": int(s.cursor),
            "pad": [
                s.pad.ones,
                s.pad.ones_carry,
                s.pad.tens,
                s.pad.tens_carry,
                s.pad.hundreds,
                s.pad.hundreds_carry,
                s.pad.thousands
            ]
        }
    
    def is_correct(self):
        return self.get_current_answer() == self.state.target

    def set_digit(self, digit):
        match self.state.cursor:
            case Cursor.ONES_OUT:
                self.state.pad.ones = digit
            case Cursor.TENS_OUT:
                self.state.pad.tens = digit
            case Cursor.HUNDREDS_OUT:
                self.state.pad.hundreds = digit
            case Cursor.THOUSANDS_OUT:
                self.state.pad.thousands = digit
            case _:
                return

    def toggle_carry(self):
        def get_toggled_carry(cur_carry):
            return 1 if cur_carry == 0 or cur_carry == EMPTY else 0
        
        match self.state.cursor:
            case Cursor.ONES_CARRY:
                self.state.pad.ones_carry = get_toggled_carry(self.state.pad.ones_carry)
            case Cursor.TENS_CARRY:
                self.state.pad.tens_carry = get_toggled_carry(self.state.pad.tens_carry)
            case Cursor.HUNDREDS_CARRY:
                self.state.pad.hundreds_carry = get_toggled_carry(self.state.pad.hundreds_carry)
            case _:
                return

    def move_cursor_left(self):
        match self.state.cursor:
            case Cursor.ONES_OUT:
                self.state.cursor = Cursor.ONES_CARRY
            case Cursor.ONES_CARRY:
                self.state.cursor = Cursor.TENS_OUT
            case Cursor.TENS_OUT:
                self.state.cursor = Cursor.TENS_CARRY
            case Cursor.TENS_CARRY:
                self.state.cursor = Cursor.HUNDREDS_OUT
            case Cursor.HUNDREDS_OUT:
                self.state.cursor = Cursor.HUNDREDS_CARRY
            case Cursor.HUNDREDS_CARRY:
                self.state.cursor = Cursor.THOUSANDS_OUT
            case _:
                return

    def move_cursor_right(self):
        match self.state.cursor:
            case Cursor.ONES_CARRY:
                self.state.cursor = Cursor.ONES_OUT
            case Cursor.TENS_OUT:
                self.state.cursor = Cursor.ONES_CARRY
            case Cursor.TENS_CARRY:
                self.state.cursor = Cursor.TENS_OUT
            case Cursor.HUNDREDS_OUT:
                self.state.cursor = Cursor.TENS_CARRY
            case Cursor.HUNDREDS_CARRY:
                self.state.cursor = Cursor.HUNDREDS_OUT
            case Cursor.THOUSANDS_OUT:
                self.state.cursor = Cursor.HUNDREDS_CARRY
            case _:
                return

    def be_done(self):
        self.state.done = True

    def initialize_state(self):
        numA = randint(0, (10**self.max_digits)-1)
        numB = randint(0, (10**self.max_digits)-1)
        self.state = EnvState(numA, numB, numA+numB)

    def reset(self):
        self.initialize_state()
        return self.get_obs()
    
    def get_current_answer(self):
        return (
            self._digit_or_zero(self.state.pad.thousands) * 1000 +
            self._digit_or_zero(self.state.pad.hundreds)  * 100 +
            self._digit_or_zero(self.state.pad.tens)      * 10 +
            self._digit_or_zero(self.state.pad.ones)
        )
    
    def _digit_or_zero(self, d):
        return 0 if d == EMPTY else d