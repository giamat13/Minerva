"""The Swift-Instruct fine-tuning set.

PROVENANCE
==========
**Where these came from:** every example below was written by hand, one at a
time, for this project. None was produced by a script, a template, or by
permuting slot values, and none was copied from another dataset. `CLAUDE.md`
forbids all three, and this file is the place that rule bites hardest - it
would have been trivial to emit ten thousand `f"What is {a} + {b}?"` rows.

**What they are meant to teach**, in priority order:

1. *The conversation format itself.* A base model has never seen a turn
   boundary. Most of the value here is teaching it to open with the assistant
   marker, emit at most one reasoning block, and stop at ``<|end|>``.
2. *When to reach for a tool.* Arithmetic and "what is today's date" cannot be
   answered from weights, and the model is taught to route them to the
   calculator and the clock rather than guess.
3. *When NOT to reach for a tool.* Roughly as many examples answer directly as
   call a tool. Without that balance a small model learns "always call the
   calculator", which is worse than no tool use at all.
4. *Saying "I don't know".* Swift was pretrained on 27 MB of 19th-century
   novels and 1987 newswire. It does not know current facts, and confident
   invention is the failure mode most worth training against.

**What was deliberately left out:** long-form answers (the context is 512
tokens), multi-tool chains (too much to learn from a set this size), and any
factual question whose answer the model cannot plausibly have learned - those
are taught as "I don't know" instead.

**Tool results are never fabricated.** An example declares the *call*, and
:func:`build_examples` executes the real tool to obtain the result that goes
into the training text. The only exceptions are the clock examples, whose
output depends on the wall clock; those carry an explicit ``result`` and are
marked ``pinned=True``.

**Size.** 185 English + 34 Hebrew. That is small on purpose: it is what one
person can actually read and stand behind, and per `CLAUDE.md` a hundred
considered examples beat a hundred thousand generated ones. It is enough to
teach a format and a routing habit, in either language. It is *not* enough to
teach knowledge, and this file does not pretend otherwise.

**Hebrew (added after v0.2.0's pretraining corpus added Hebrew).** 34
examples, section 8 below, deliberately smaller and more direct-answer-heavy
than the English set - see that section's own comment for why. This is a
first, measured round, not a claim that Hebrew is as well covered as English:
it is not, and the held-out numbers in `docs/TRAINING.md` say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..messages import Message, Role, ToolCall, assistant, user

__all__ = ["INSTRUCT_EXAMPLES", "InstructExample", "build_examples"]


@dataclass(frozen=True, slots=True)
class InstructExample:
    """One hand-written training conversation."""

    user: str
    answer: str
    think: str | None = None
    call: tuple[str, dict[str, Any]] | None = None
    result: str | None = None
    """Only for tools whose output is not reproducible (the clock)."""
    pinned: bool = False
    """True when ``result`` is supplied rather than computed by running the tool."""
    history: tuple[tuple[str, str], ...] = field(default=())
    """Earlier (user, assistant) turns, for the multi-turn examples."""


# ---------------------------------------------------------------------------
# 1. Arithmetic that must go to the calculator.
#    Phrasings are varied on purpose: "work out", "how much is", "compute",
#    bare expressions, word problems. A model trained only on "What is X + Y?"
#    learns the sentence, not the task.
# ---------------------------------------------------------------------------

_CALCULATOR: tuple[InstructExample, ...] = (
    InstructExample(
        user="What is 17 times 43?",
        call=("calculate", {"expression": "17 * 43"}),
        answer="17 times 43 is 731.",
    ),
    InstructExample(
        user="Work out 2847 divided by 13.",
        call=("calculate", {"expression": "2847 / 13"}),
        answer="2847 divided by 13 is 219.",
    ),
    InstructExample(
        user="How much is 1999 plus 2024?",
        call=("calculate", {"expression": "1999 + 2024"}),
        answer="That comes to 4023.",
    ),
    InstructExample(
        user="Compute 144 minus 89.",
        call=("calculate", {"expression": "144 - 89"}),
        answer="144 minus 89 is 55.",
    ),
    InstructExample(
        user="9 squared",
        call=("calculate", {"expression": "9 ** 2"}),
        answer="9 squared is 81.",
    ),
    InstructExample(
        user="What's the square root of 576?",
        call=("calculate", {"expression": "sqrt(576)"}),
        answer="The square root of 576 is 24.",
    ),
    InstructExample(
        user="Give me 15% of 240.",
        call=("calculate", {"expression": "240 * 0.15"}),
        answer="15% of 240 is 36.",
    ),
    InstructExample(
        user="I need the total of 12.50, 8.75 and 4.20.",
        call=("calculate", {"expression": "12.50 + 8.75 + 4.20"}),
        answer="The total is 25.45.",
    ),
    InstructExample(
        user="Divide 1000 by 7 and tell me the answer.",
        call=("calculate", {"expression": "1000 / 7"}),
        answer="1000 divided by 7 is about 142.86.",
    ),
    InstructExample(
        user="What is 2 to the power of 16?",
        call=("calculate", {"expression": "2 ** 16"}),
        answer="2 to the power of 16 is 65536.",
    ),
    InstructExample(
        user="Add up 45, 67, 89 and 23.",
        call=("calculate", {"expression": "45 + 67 + 89 + 23"}),
        answer="They add up to 224.",
    ),
    InstructExample(
        user="How many seconds are in a day?",
        call=("calculate", {"expression": "24 * 60 * 60"}),
        answer="There are 86400 seconds in a day.",
    ),
    InstructExample(
        user="A book costs 34 shekels. How much for 7 copies?",
        call=("calculate", {"expression": "34 * 7"}),
        answer="Seven copies cost 238 shekels.",
    ),
    InstructExample(
        user="Split 8450 between 25 people.",
        call=("calculate", {"expression": "8450 / 25"}),
        answer="Each person gets 338.",
    ),
    InstructExample(
        user="What is 6 factorial?",
        call=("calculate", {"expression": "factorial(6)"}),
        answer="6 factorial is 720.",
    ),
    InstructExample(
        user="Convert 100 degrees Celsius to Fahrenheit.",
        call=("calculate", {"expression": "100 * 9 / 5 + 32"}),
        answer="100 degrees Celsius is 212 degrees Fahrenheit.",
    ),
    InstructExample(
        user="What is the remainder when 100 is divided by 7?",
        call=("calculate", {"expression": "100 % 7"}),
        answer="The remainder is 2.",
    ),
    InstructExample(
        user="Multiply 125 by 8.",
        call=("calculate", {"expression": "125 * 8"}),
        answer="125 times 8 is 1000.",
    ),
    InstructExample(
        user="If I earn 4200 a month, what is that a year?",
        call=("calculate", {"expression": "4200 * 12"}),
        answer="That is 50400 a year.",
    ),
    InstructExample(
        user="What is the average of 12, 18 and 27?",
        call=("calculate", {"expression": "(12 + 18 + 27) / 3"}),
        answer="The average is 19.",
    ),
    InstructExample(
        user="Take 20 percent off 85.",
        call=("calculate", {"expression": "85 * 0.8"}),
        answer="With 20 percent off, it comes to 68.",
    ),
    InstructExample(
        user="How much is a third of 291?",
        call=("calculate", {"expression": "291 / 3"}),
        answer="A third of 291 is 97.",
    ),
    InstructExample(
        user="What is 45 plus 55 minus 30?",
        call=("calculate", {"expression": "45 + 55 - 30"}),
        answer="That is 70.",
    ),
    InstructExample(
        user="Calculate the area of a circle with radius 5.",
        call=("calculate", {"expression": "pi * 5 ** 2"}),
        answer="The area is about 78.54.",
    ),
    InstructExample(
        user="What is 3 cubed?",
        call=("calculate", {"expression": "3 ** 3"}),
        answer="3 cubed is 27.",
    ),
    InstructExample(
        user="Sum the numbers from 1 to 100.",
        call=("calculate", {"expression": "100 * 101 / 2"}),
        answer="The sum is 5050.",
    ),
    InstructExample(
        user="How many minutes in three and a half hours?",
        call=("calculate", {"expression": "3.5 * 60"}),
        answer="That is 210 minutes.",
    ),
    InstructExample(
        user="What is 987 times 654?",
        call=("calculate", {"expression": "987 * 654"}),
        answer="987 times 654 is 645498.",
    ),
    InstructExample(
        user="Is 91 divisible by 7?",
        call=("calculate", {"expression": "91 / 7"}),
        answer="Yes. 91 divided by 7 is exactly 13.",
    ),
    InstructExample(
        user="Give me the greatest common divisor of 48 and 180.",
        call=("calculate", {"expression": "gcd(48, 180)"}),
        answer="The greatest common divisor is 12.",
    ),
)


# ---------------------------------------------------------------------------
# 2. Arithmetic reached through a reasoning step. Same tool, but the model is
#    taught to say why before acting - which is what makes a thinking level
#    mean something.
# ---------------------------------------------------------------------------

_CALCULATOR_WITH_THINKING: tuple[InstructExample, ...] = (
    InstructExample(
        user="A shop sells pens at 7 for 21 shekels. What do 12 pens cost?",
        think="One pen is 21 divided by 7, which is 3. So twelve pens is 12 times 3.",
        call=("calculate", {"expression": "12 * (21 / 7)"}),
        answer="Twelve pens cost 36 shekels.",
    ),
    InstructExample(
        user="I bought 3 shirts at 89 each and paid with 300. What is my change?",
        think="Three shirts is 3 times 89. The change is 300 minus that.",
        call=("calculate", {"expression": "300 - 3 * 89"}),
        answer="Your change is 33.",
    ),
    InstructExample(
        user="A train covers 240 km in 3 hours. What is its speed?",
        think="Speed is distance divided by time, so 240 divided by 3.",
        call=("calculate", {"expression": "240 / 3"}),
        answer="Its speed is 80 km per hour.",
    ),
    InstructExample(
        user="If a rectangle is 14 by 9, what is its perimeter?",
        think="Perimeter is twice the sum of the two sides.",
        call=("calculate", {"expression": "2 * (14 + 9)"}),
        answer="The perimeter is 46.",
    ),
    InstructExample(
        user="A recipe for 4 people needs 300g of flour. How much for 10?",
        think="Per person it is 300 divided by 4. Multiply that by 10.",
        call=("calculate", {"expression": "300 / 4 * 10"}),
        answer="You need 750g of flour.",
    ),
    InstructExample(
        user="I saved 250 a month for two years. How much is that?",
        think="Two years is 24 months, so 250 times 24.",
        call=("calculate", {"expression": "250 * 24"}),
        answer="You saved 6000.",
    ),
    InstructExample(
        user="A 5000 loan at 4 percent simple interest for one year. What is the interest?",
        think="Simple interest for one year is the principal times the rate.",
        call=("calculate", {"expression": "5000 * 0.04"}),
        answer="The interest is 200.",
    ),
    InstructExample(
        user="Sixteen people share a bill of 1200 equally. What does each pay?",
        think="Divide the bill by the number of people.",
        call=("calculate", {"expression": "1200 / 16"}),
        answer="Each person pays 75.",
    ),
    InstructExample(
        user="A tank holds 450 litres and is two thirds full. How many litres is that?",
        think="Two thirds of 450.",
        call=("calculate", {"expression": "450 * 2 / 3"}),
        answer="There are 300 litres in the tank.",
    ),
    InstructExample(
        user="What is the diagonal of a 3 by 4 rectangle?",
        think="That is the hypotenuse of a right triangle with sides 3 and 4.",
        call=("calculate", {"expression": "hypot(3, 4)"}),
        answer="The diagonal is 5.",
    ),
    InstructExample(
        user="A car uses 7 litres per 100 km. How much for a 350 km trip?",
        think="Seven litres per hundred kilometres, over 3.5 hundreds.",
        call=("calculate", {"expression": "7 * 350 / 100"}),
        answer="The trip uses 24.5 litres.",
    ),
    InstructExample(
        user="I have 1440 minutes. How many hours is that?",
        think="Sixty minutes to the hour, so divide by 60.",
        call=("calculate", {"expression": "1440 / 60"}),
        answer="That is 24 hours.",
    ),
)


# ---------------------------------------------------------------------------
# 3. The clock. Results are pinned because the real output depends on the wall
#    clock and could not be reproduced by a later build.
# ---------------------------------------------------------------------------

_CLOCK: tuple[InstructExample, ...] = (
    InstructExample(
        user="What is today's date?",
        call=("current_time", {"timezone": "UTC"}),
        result="2026-03-14 09:41:02 (Saturday) in UTC [UTC+00:00]",
        pinned=True,
        answer="Today is Saturday, 14 March 2026.",
    ),
    InstructExample(
        user="What time is it in Jerusalem?",
        call=("current_time", {"timezone": "Asia/Jerusalem"}),
        result="2026-03-14 11:41:02 (Saturday) in Asia/Jerusalem [UTC+02:00]",
        pinned=True,
        answer="It is 11:41 in Jerusalem.",
    ),
    InstructExample(
        user="Tell me the current time in New York.",
        call=("current_time", {"timezone": "America/New_York"}),
        result="2026-03-14 05:41:02 (Saturday) in America/New_York [UTC-04:00]",
        pinned=True,
        answer="It is 05:41 in New York.",
    ),
    InstructExample(
        user="What day of the week is it?",
        call=("current_time", {"timezone": "UTC"}),
        result="2026-03-14 09:41:02 (Saturday) in UTC [UTC+00:00]",
        pinned=True,
        answer="It is Saturday.",
    ),
    InstructExample(
        user="What is the time in London right now?",
        call=("current_time", {"timezone": "Europe/London"}),
        result="2026-03-14 09:41:02 (Saturday) in Europe/London [UTC+00:00]",
        pinned=True,
        answer="It is 09:41 in London.",
    ),
    InstructExample(
        user="How many days are there between 2026-01-01 and 2026-03-01?",
        call=("days_between", {"start_date": "2026-01-01", "end_date": "2026-03-01"}),
        answer="There are 59 days between those dates.",
    ),
    InstructExample(
        user="Count the days from 2024-02-01 to 2024-03-01.",
        call=("days_between", {"start_date": "2024-02-01", "end_date": "2024-03-01"}),
        answer="That is 29 days, because 2024 was a leap year.",
    ),
    InstructExample(
        user="How long is it from 2026-06-15 until 2026-12-25?",
        call=("days_between", {"start_date": "2026-06-15", "end_date": "2026-12-25"}),
        answer="It is 193 days.",
    ),
    InstructExample(
        user="Days between 2025-11-01 and 2026-01-01?",
        call=("days_between", {"start_date": "2025-11-01", "end_date": "2026-01-01"}),
        answer="There are 61 days.",
    ),
    InstructExample(
        user="I need the number of days from 2026-03-14 to 2026-04-30.",
        think="Both dates are given in full, so I can measure the gap directly.",
        call=("days_between", {"start_date": "2026-03-14", "end_date": "2026-04-30"}),
        answer="That is 47 days.",
    ),
    InstructExample(
        user="What is the date in Tokyo?",
        think="The date depends on the time zone, so I should ask for Tokyo specifically.",
        call=("current_time", {"timezone": "Asia/Tokyo"}),
        result="2026-03-14 18:41:02 (Saturday) in Asia/Tokyo [UTC+09:00]",
        pinned=True,
        answer="In Tokyo it is Saturday, 14 March 2026.",
    ),
)


# ---------------------------------------------------------------------------
# 4. Questions that need NO tool. There are deliberately about as many of these
#    as there are tool examples: a model taught only tool use calls the
#    calculator to say hello.
# ---------------------------------------------------------------------------

_NO_TOOL: tuple[InstructExample, ...] = (
    InstructExample(user="Hello.", answer="Hello. How can I help?"),
    InstructExample(user="Hi there.", answer="Hello. What would you like to do?"),
    InstructExample(user="Good morning.", answer="Good morning."),
    InstructExample(user="Thank you.", answer="You are welcome."),
    InstructExample(user="Thanks, that helped.", answer="Glad to hear it."),
    InstructExample(user="Goodbye.", answer="Goodbye."),
    InstructExample(
        user="Who are you?",
        answer="I am Swift, the smallest model in the Minerva family.",
    ),
    InstructExample(user="What is your name?", answer="My name is Swift."),
    InstructExample(
        user="What can you do?",
        answer=(
            "I can answer short questions, and I can use a calculator and a "
            "clock when a question needs one."
        ),
    ),
    InstructExample(
        user="Are you a large model?",
        answer="No. I am a small model, about ten million parameters.",
    ),
    InstructExample(
        user="Can you write me a long essay?",
        answer="No. I am small and my context is short, so I answer briefly.",
    ),
    InstructExample(user="Repeat the word banana.", answer="banana"),
    InstructExample(user="Say the word ready.", answer="ready"),
    InstructExample(
        user="Reply with exactly one word: yes or no. Is water wet?",
        answer="Yes.",
    ),
    InstructExample(user="Count from one to five.", answer="One, two, three, four, five."),
    InstructExample(
        user="List the days of the week.",
        answer="Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.",
    ),
    InstructExample(user="Name the four seasons.", answer="Spring, summer, autumn and winter."),
    InstructExample(user="How many days are in a week?", answer="Seven."),
    InstructExample(user="How many months are in a year?", answer="Twelve."),
    InstructExample(user="How many hours are in a day?", answer="Twenty-four."),
    InstructExample(user="What comes after Wednesday?", answer="Thursday."),
    InstructExample(user="Spell the word bird backwards.", answer="drib"),
    InstructExample(user="Make this upper case: hello world", answer="HELLO WORLD"),
    InstructExample(user="Make this lower case: STOP SHOUTING", answer="stop shouting"),
    InstructExample(
        user="Rewrite this more politely: give me the report.",
        answer="Could you please send me the report?",
    ),
    InstructExample(
        user="Shorten this: I would like to request that you consider my application.",
        answer="Please consider my application.",
    ),
    InstructExample(
        user="Turn this into a question: the train leaves at six.",
        answer="Does the train leave at six?",
    ),
    InstructExample(
        user="Correct the spelling: I recieved the letter.",
        answer="I received the letter.",
    ),
    InstructExample(
        user="Give me a shorter word for 'commence'.",
        answer="Begin.",
    ),
    InstructExample(
        user="What is the opposite of 'early'?",
        answer="Late.",
    ),
    InstructExample(
        user="Finish this sentence: the sun rose over the",
        answer="the sun rose over the hills.",
    ),
    InstructExample(
        user="Write one sentence about a bird.",
        answer="The swift spends almost its whole life in the air.",
    ),
    InstructExample(
        user="Give me a title for a story about the sea.",
        answer="The Long Water.",
    ),
    InstructExample(
        user="Is 'run' a noun or a verb?",
        answer="It can be either, but it is most often a verb.",
    ),
    InstructExample(
        user="How do you spell the word 'necessary'?",
        answer="n-e-c-e-s-s-a-r-y.",
    ),
)


# ---------------------------------------------------------------------------
# 5. Reasoning without a tool. Teaches that a thinking block does not imply
#    a tool call.
# ---------------------------------------------------------------------------

_THINKING_NO_TOOL: tuple[InstructExample, ...] = (
    InstructExample(
        user="If today is Wednesday, what day is it in two days?",
        think="Wednesday plus one is Thursday, plus one more is Friday.",
        answer="It will be Friday.",
    ),
    InstructExample(
        user="A man has two coins that add to 30, and one of them is not a 20. What are they?",
        think="Only one of them is not a 20, so the other one can be.",
        answer="A 20 and a 10. The 10 is the one that is not a 20.",
    ),
    InstructExample(
        user="Is the word 'level' the same backwards?",
        think="Reading it backwards gives l-e-v-e-l, which is the same.",
        answer="Yes, it is a palindrome.",
    ),
    InstructExample(
        user="Which is longer, a metre or a yard?",
        think="A yard is about 0.91 metres, so the metre is longer.",
        answer="A metre is longer.",
    ),
    InstructExample(
        user="If all swifts are birds and all birds have feathers, do swifts have feathers?",
        think="This follows directly from the two statements.",
        answer="Yes, they do.",
    ),
    InstructExample(
        user="I have three apples and eat one. Then I buy two. How many do I have?",
        think="Three minus one is two, and two more makes four. Small enough to do in my head.",
        answer="You have four apples.",
    ),
    InstructExample(
        user="Put these in order: Friday, Monday, Wednesday.",
        think="Monday comes first in the week, then Wednesday, then Friday.",
        answer="Monday, Wednesday, Friday.",
    ),
    InstructExample(
        user="Which is heavier, a kilogram of feathers or a kilogram of iron?",
        think="Both are one kilogram; the difference is volume, not mass.",
        answer="Neither. They weigh the same.",
    ),
    InstructExample(
        user="If a shirt was 100 and is now 80, did the price rise or fall?",
        think="Eighty is less than a hundred, so it fell.",
        answer="It fell.",
    ),
    InstructExample(
        user="Sort these words alphabetically: pear, apple, orange.",
        think="Apple begins with a, orange with o, pear with p.",
        answer="apple, orange, pear.",
    ),
    InstructExample(
        user="Is February always 28 days?",
        think="Leap years add a day, and they come every four years.",
        answer="No. In a leap year it has 29 days.",
    ),
    InstructExample(
        user="Is midnight the start or the end of a day?",
        think="By convention 00:00 begins the new day.",
        answer="It is the start of a day.",
    ),
)


# ---------------------------------------------------------------------------
# 6. Honest ignorance. The most important category for a model this small: it
#    was pretrained on 19th-century novels and 1987 newswire, and confident
#    invention is its worst failure mode.
# ---------------------------------------------------------------------------

_UNKNOWN: tuple[InstructExample, ...] = (
    InstructExample(
        user="Who won the World Cup in 2022?",
        answer="I do not know. I am a small model and I was not trained on recent events.",
    ),
    InstructExample(
        user="What is the population of Brazil?",
        answer="I do not know that figure, and I would rather not guess.",
    ),
    InstructExample(
        user="What is the current price of gold?",
        answer="I cannot know that. I have no access to live prices.",
    ),
    InstructExample(
        user="Who is the prime minister right now?",
        answer="I do not know. My training data does not cover current office holders.",
    ),
    InstructExample(
        user="What is the weather today?",
        answer="I cannot tell you. I have no weather data.",
    ),
    InstructExample(
        user="Summarise the news from this morning.",
        answer="I cannot. I have no access to today's news.",
    ),
    InstructExample(
        user="What is my name?",
        answer="I do not know your name. You have not told me.",
    ),
    InstructExample(
        user="What did we talk about last week?",
        answer="I do not remember previous conversations.",
    ),
    InstructExample(
        user="How tall is the tallest building in the world?",
        answer="I am not confident of that figure, so I would rather not state one.",
    ),
    InstructExample(
        user="Explain quantum chromodynamics in detail.",
        answer="That is beyond what I can explain reliably. I am a very small model.",
    ),
    InstructExample(
        user="Write a working web server in Python.",
        answer="I cannot do that reliably. I was not trained on code.",
    ),
    InstructExample(
        user="Translate this into Japanese: good evening.",
        answer="I cannot translate reliably. I was trained almost entirely on English.",
    ),
    InstructExample(
        user="What will happen tomorrow?",
        answer="I have no way to know that.",
    ),
    InstructExample(
        user="Give me medical advice about my headache.",
        answer="I am not able to give medical advice. Please ask a doctor.",
    ),
    InstructExample(
        user="What is the capital of France?",
        think="This is a fact I may have seen, but I am a very small model and I am not confident.",
        answer="I believe it is Paris, though I am a small model and can be wrong.",
    ),
)


# ---------------------------------------------------------------------------
# 7. Multi-turn. Teaches that the format repeats and that earlier turns are
#    context, not something to answer again.
# ---------------------------------------------------------------------------

_MULTI_TURN: tuple[InstructExample, ...] = (
    InstructExample(
        history=(("Hello.", "Hello. How can I help?"),),
        user="What is 12 times 12?",
        call=("calculate", {"expression": "12 * 12"}),
        answer="12 times 12 is 144.",
    ),
    InstructExample(
        history=(("What is 10 plus 5?", "10 plus 5 is 15."),),
        user="Now double it.",
        think="The previous answer was 15, so I need 15 times 2.",
        call=("calculate", {"expression": "15 * 2"}),
        answer="Doubled, that is 30.",
    ),
    InstructExample(
        history=(("My name is Dana.", "Nice to meet you, Dana."),),
        user="What is my name?",
        answer="Your name is Dana.",
    ),
    InstructExample(
        history=(("Who are you?", "I am Swift, the smallest model in the Minerva family."),),
        user="Who made you?",
        answer="I was trained by Minerva.",
    ),
    InstructExample(
        history=(("What is 100 divided by 4?", "100 divided by 4 is 25."),),
        user="And what is that times 3?",
        call=("calculate", {"expression": "25 * 3"}),
        answer="That is 75.",
    ),
    InstructExample(
        history=(
            ("I am planning a trip.", "That sounds good. Where are you going?"),
            ("To London.", "Enjoy London."),
        ),
        user="What time is it there?",
        call=("current_time", {"timezone": "Europe/London"}),
        result="2026-03-14 09:41:02 (Saturday) in Europe/London [UTC+00:00]",
        pinned=True,
        answer="It is 09:41 in London.",
    ),
    InstructExample(
        history=(("Say the word ready.", "ready"),),
        user="Again please.",
        answer="ready",
    ),
    InstructExample(
        history=(("What is the square root of 81?", "The square root of 81 is 9."),),
        user="Is that a whole number?",
        think="Nine has no fractional part.",
        answer="Yes, 9 is a whole number.",
    ),
)


# ---------------------------------------------------------------------------
# 1b. More two-operand arithmetic, added after the first fine-tune measured
#     argument accuracy at 11%. The model was routing correctly and picking the
#     right tool, but copying only the FIRST number out of the question - it
#     answered "23 times 19" with `23 * 2`.
#
#     These target that specific failure: both operands are always restated in
#     the expression in the order they appear, and the digit lengths are varied
#     (1, 2, 3 and 4 digits) so the habit cannot latch onto one shape. Written
#     by hand like the rest - the temptation to emit a loop over number pairs
#     here is exactly what CLAUDE.md forbids, and it would teach the sentence
#     rather than the skill.
# ---------------------------------------------------------------------------

_CALCULATOR_OPERAND_COPYING: tuple[InstructExample, ...] = (
    InstructExample(
        user="What is 8 times 6?",
        call=("calculate", {"expression": "8 * 6"}),
        answer="8 times 6 is 48.",
    ),
    InstructExample(
        user="What is 4 plus 9?",
        call=("calculate", {"expression": "4 + 9"}),
        answer="4 plus 9 is 13.",
    ),
    InstructExample(
        user="What is 15 minus 7?",
        call=("calculate", {"expression": "15 - 7"}),
        answer="15 minus 7 is 8.",
    ),
    InstructExample(
        user="What is 36 divided by 4?",
        call=("calculate", {"expression": "36 / 4"}),
        answer="36 divided by 4 is 9.",
    ),
    InstructExample(
        user="What is 27 times 3?",
        call=("calculate", {"expression": "27 * 3"}),
        answer="27 times 3 is 81.",
    ),
    InstructExample(
        user="What is 52 plus 38?",
        call=("calculate", {"expression": "52 + 38"}),
        answer="52 plus 38 is 90.",
    ),
    InstructExample(
        user="What is 84 minus 29?",
        call=("calculate", {"expression": "84 - 29"}),
        answer="84 minus 29 is 55.",
    ),
    InstructExample(
        user="What is 63 divided by 9?",
        call=("calculate", {"expression": "63 / 9"}),
        answer="63 divided by 9 is 7.",
    ),
    InstructExample(
        user="What is 19 times 4?",
        call=("calculate", {"expression": "19 * 4"}),
        answer="19 times 4 is 76.",
    ),
    InstructExample(
        user="What is 46 plus 17?",
        call=("calculate", {"expression": "46 + 17"}),
        answer="46 plus 17 is 63.",
    ),
    InstructExample(
        user="What is 73 minus 48?",
        call=("calculate", {"expression": "73 - 48"}),
        answer="73 minus 48 is 25.",
    ),
    InstructExample(
        user="What is 96 divided by 8?",
        call=("calculate", {"expression": "96 / 8"}),
        answer="96 divided by 8 is 12.",
    ),
    InstructExample(
        user="What is 34 times 21?",
        call=("calculate", {"expression": "34 * 21"}),
        answer="34 times 21 is 714.",
    ),
    InstructExample(
        user="What is 128 plus 256?",
        call=("calculate", {"expression": "128 + 256"}),
        answer="128 plus 256 is 384.",
    ),
    InstructExample(
        user="What is 405 minus 178?",
        call=("calculate", {"expression": "405 - 178"}),
        answer="405 minus 178 is 227.",
    ),
    InstructExample(
        user="What is 512 divided by 16?",
        call=("calculate", {"expression": "512 / 16"}),
        answer="512 divided by 16 is 32.",
    ),
    InstructExample(
        user="What is 306 times 5?",
        call=("calculate", {"expression": "306 * 5"}),
        answer="306 times 5 is 1530.",
    ),
    InstructExample(
        user="What is 749 plus 251?",
        call=("calculate", {"expression": "749 + 251"}),
        answer="749 plus 251 is 1000.",
    ),
    InstructExample(
        user="What is 620 minus 95?",
        call=("calculate", {"expression": "620 - 95"}),
        answer="620 minus 95 is 525.",
    ),
    InstructExample(
        user="What is 888 divided by 8?",
        call=("calculate", {"expression": "888 / 8"}),
        answer="888 divided by 8 is 111.",
    ),
    InstructExample(
        user="What is 1024 plus 976?",
        call=("calculate", {"expression": "1024 + 976"}),
        answer="1024 plus 976 is 2000.",
    ),
    InstructExample(
        user="What is 3600 divided by 60?",
        call=("calculate", {"expression": "3600 / 60"}),
        answer="3600 divided by 60 is 60.",
    ),
    InstructExample(
        user="What is 1250 minus 375?",
        call=("calculate", {"expression": "1250 - 375"}),
        answer="1250 minus 375 is 875.",
    ),
    InstructExample(
        user="What is 2500 times 4?",
        call=("calculate", {"expression": "2500 * 4"}),
        answer="2500 times 4 is 10000.",
    ),
    InstructExample(
        user="Multiply 7 and 13.",
        call=("calculate", {"expression": "7 * 13"}),
        answer="7 times 13 is 91.",
    ),
    InstructExample(
        user="Add 240 to 360.",
        call=("calculate", {"expression": "240 + 360"}),
        answer="That is 600.",
    ),
    InstructExample(
        user="Take 45 away from 120.",
        call=("calculate", {"expression": "120 - 45"}),
        answer="That leaves 75.",
    ),
    InstructExample(
        user="Divide 550 by 11.",
        call=("calculate", {"expression": "550 / 11"}),
        answer="550 divided by 11 is 50.",
    ),
    InstructExample(
        user="How much is 68 times 9?",
        call=("calculate", {"expression": "68 * 9"}),
        answer="68 times 9 is 612.",
    ),
    InstructExample(
        user="How much is 900 minus 149?",
        call=("calculate", {"expression": "900 - 149"}),
        answer="900 minus 149 is 751.",
    ),
    InstructExample(
        user="Compute 14 times 14.",
        call=("calculate", {"expression": "14 * 14"}),
        answer="14 times 14 is 196.",
    ),
    InstructExample(
        user="Compute 81 divided by 27.",
        call=("calculate", {"expression": "81 / 27"}),
        answer="81 divided by 27 is 3.",
    ),
    InstructExample(
        user="Work out 220 plus 780.",
        call=("calculate", {"expression": "220 + 780"}),
        answer="That comes to 1000.",
    ),
    InstructExample(
        user="Work out 57 times 6.",
        call=("calculate", {"expression": "57 * 6"}),
        answer="57 times 6 is 342.",
    ),
)


# ---------------------------------------------------------------------------
# 6b. More direct answers and more refusals, added to REBALANCE the set.
#
#     Adding 34 arithmetic examples in 1b lifted argument accuracy from 11% to
#     28% - and dropped honest refusal from 50% to 17% and routing from 91% to
#     85%. The set had tipped to 91 tool examples against 66 without, and the
#     model learned "reach for the calculator". That is precisely the failure
#     this file's own docstring warns about, and it showed up in the numbers
#     within one run.
#
#     These restore the balance. The measured effect of both rounds is in
#     docs/TRAINING.md, including the regression, because a tuning decision
#     that is only reported when it worked is not a report.
# ---------------------------------------------------------------------------

_REBALANCE: tuple[InstructExample, ...] = (
    # More things it should decline rather than invent.
    InstructExample(
        user="What is the exchange rate for the euro?",
        answer="I do not know. I have no access to live rates.",
    ),
    InstructExample(
        user="Who is the CEO of that company?",
        answer="I do not know, and I would rather not guess at a name.",
    ),
    InstructExample(
        user="What is the score in the match?",
        answer="I cannot know that. I have no access to live results.",
    ),
    InstructExample(
        user="Recommend a restaurant near me.",
        answer="I cannot. I do not know where you are.",
    ),
    InstructExample(
        user="What is in my calendar today?",
        answer="I have no access to your calendar.",
    ),
    InstructExample(
        user="Read this file for me.",
        answer="I cannot read files. I can only reply to what you type.",
    ),
    InstructExample(
        user="Search the web for me.",
        answer="I cannot search the web. I have no network access.",
    ),
    InstructExample(
        user="Tell me about a book published last year.",
        answer="I do not know. My training data does not cover recent books.",
    ),
    InstructExample(
        user="How old am I?",
        answer="I do not know your age. You have not told me.",
    ),
    InstructExample(
        user="What is the best programming language?",
        answer="I do not have a reliable opinion on that.",
    ),
    InstructExample(
        user="Give me legal advice about my contract.",
        answer="I am not able to give legal advice. Please ask a lawyer.",
    ),
    InstructExample(
        user="Explain how a jet engine works in detail.",
        answer="I cannot explain that reliably. I am a very small model.",
    ),
    # More questions that look numeric but need no tool at all.
    InstructExample(user="How many letters are in the word cat?", answer="Three."),
    InstructExample(user="How many sides does a triangle have?", answer="Three."),
    InstructExample(user="How many legs does a bird have?", answer="Two."),
    InstructExample(user="How many minutes are in an hour?", answer="Sixty."),
    InstructExample(user="What number comes after nine?", answer="Ten."),
    InstructExample(user="Is seven an odd number?", answer="Yes, seven is odd."),
    # More ordinary conversation.
    InstructExample(user="How are you?", answer="I am working. What do you need?"),
    InstructExample(user="Are you there?", answer="Yes, I am here."),
    InstructExample(user="Never mind.", answer="All right."),
    InstructExample(user="That is wrong.", answer="Sorry. Could you tell me what it should be?"),
    InstructExample(user="Try again.", answer="Could you say what you would like me to try?"),
    InstructExample(user="Please be brief.", answer="Understood. I will keep it short."),
    InstructExample(user="Stop.", answer="Stopped."),
    InstructExample(user="Can you help me?", answer="Yes. What do you need?"),
    InstructExample(
        user="Write a single short sentence about the sea.",
        answer="The sea was grey and very wide.",
    ),
    InstructExample(
        user="Give me one word for a small fast bird.",
        answer="Swift.",
    ),
)


# ---------------------------------------------------------------------------
# 8. Hebrew. Swift v0.2.0's pretraining corpus added curated Hebrew literature
#    (docs/TRAINING.md, S1) - the base model can write grammatical Hebrew, but
#    had never seen a Hebrew turn boundary, a Hebrew tool call, or a Hebrew
#    refusal. These teach the same three things S1-6 teach in English: open
#    with the assistant marker and stop at <|end|>, route arithmetic and dates
#    to a tool without being told to, and say "I do not know" in Hebrew rather
#    than invent an answer just because the language changed.
#
#    Register is ordinary spoken Hebrew, not the corpus's literary register -
#    matching every English example here, which is conversational, not
#    Austen. Tool names and arguments stay in their real form (calculate's
#    expressions are math notation, current_time's zones are IANA names,
#    days_between's dates are ISO) - none of that is language-dependent, only
#    the surrounding sentence is.
#
#    Deliberately smaller and more direct-answer-heavy than a first English
#    round would be: S6b's own history shows a 34-example, tool-heavy addition
#    swinging honest refusal from 50% to 17% in a single round on an *already
#    well-balanced* set. Adding a second language to an already-thin
#    185-example budget is a second version of exactly that risk, so this
#    round leans away from tool calls rather than toward them (14 tool-routing
#    examples against 20 that answer directly or decline) and is written to be
#    measured before any larger Hebrew round is written on top of it.
# ---------------------------------------------------------------------------

_HEBREW: tuple[InstructExample, ...] = (
    # -- Calculator, in Hebrew --------------------------------------------
    InstructExample(
        user="כמה זה 17 כפול 43?",
        call=("calculate", {"expression": "17 * 43"}),
        answer="17 כפול 43 זה 731.",
    ),
    InstructExample(
        user="חשב 2847 חלקי 13.",
        call=("calculate", {"expression": "2847 / 13"}),
        answer="2847 חלקי 13 זה 219.",
    ),
    InstructExample(
        user="כמה זה 1999 ועוד 2024?",
        call=("calculate", {"expression": "1999 + 2024"}),
        answer="יוצא 4023.",
    ),
    InstructExample(
        user="מה זה 144 פחות 89?",
        call=("calculate", {"expression": "144 - 89"}),
        answer="144 פחות 89 זה 55.",
    ),
    InstructExample(
        user="9 בריבוע.",
        call=("calculate", {"expression": "9 ** 2"}),
        answer="9 בריבוע זה 81.",
    ),
    InstructExample(
        user="מה השורש הריבועי של 576?",
        call=("calculate", {"expression": "sqrt(576)"}),
        answer="השורש הריבועי של 576 הוא 24.",
    ),
    InstructExample(
        user="תן לי 15 אחוז מ-240.",
        call=("calculate", {"expression": "240 * 0.15"}),
        answer="15 אחוז מ-240 זה 36.",
    ),
    InstructExample(
        user="ספר עולה 34 שקלים. כמה יעלו 7 עותקים?",
        call=("calculate", {"expression": "34 * 7"}),
        answer="שבעה עותקים יעלו 238 שקלים.",
    ),
    InstructExample(
        user="כמה זה 2 בחזקת 10?",
        call=("calculate", {"expression": "2 ** 10"}),
        answer="2 בחזקת 10 זה 1024.",
    ),
    InstructExample(
        user="כפול 125 ב-8.",
        call=("calculate", {"expression": "125 * 8"}),
        answer="125 כפול 8 זה 1000.",
    ),
    # -- Clock and dates, in Hebrew -----------------------------------------
    InstructExample(
        user="מה התאריך היום?",
        call=("current_time", {"timezone": "UTC"}),
        result="2026-03-14 09:41:02 (Saturday) in UTC [UTC+00:00]",
        pinned=True,
        answer="היום יום שבת, 14 במרץ 2026.",
    ),
    InstructExample(
        user="מה השעה בירושלים?",
        call=("current_time", {"timezone": "Asia/Jerusalem"}),
        result="2026-03-14 11:41:02 (Saturday) in Asia/Jerusalem [UTC+02:00]",
        pinned=True,
        answer="השעה בירושלים היא 11:41.",
    ),
    InstructExample(
        user="איזה יום בשבוע היום?",
        call=("current_time", {"timezone": "UTC"}),
        result="2026-03-14 09:41:02 (Saturday) in UTC [UTC+00:00]",
        pinned=True,
        answer="היום יום שבת.",
    ),
    InstructExample(
        user="כמה ימים יש בין 2026-01-01 ל-2026-03-01?",
        call=("days_between", {"start_date": "2026-01-01", "end_date": "2026-03-01"}),
        answer="יש 59 ימים בין התאריכים האלה.",
    ),
    # -- Ordinary conversation, no tool --------------------------------------
    InstructExample(user="שלום.", answer="שלום. איך אפשר לעזור?"),
    InstructExample(user="בוקר טוב.", answer="בוקר טוב."),
    InstructExample(user="תודה רבה.", answer="בבקשה."),
    InstructExample(user="להתראות.", answer="להתראות."),
    InstructExample(
        user="מי אתה?",
        answer="אני Swift, המודל הקטן במשפחת Minerva.",
    ),
    InstructExample(user="מה השם שלך?", answer="השם שלי הוא Swift."),
    InstructExample(
        user="מה אתה יכול לעשות?",
        answer="אני יכול לענות על שאלות קצרות, ולהשתמש במחשבון ובשעון כשצריך.",
    ),
    InstructExample(
        user="האם אתה מודל גדול?",
        answer="לא. אני מודל קטן, בערך עשרה מיליון פרמטרים.",
    ),
    InstructExample(
        user="תכתוב לי חיבור ארוך.",
        answer="לא. אני קטן וההקשר שלי קצר, אז אני עונה בקצרה.",
    ),
    InstructExample(user="כמה ימים יש בשבוע?", answer="שבעה."),
    InstructExample(user="כמה חודשים יש בשנה?", answer="שנים עשר."),
    InstructExample(user="מה בא אחרי יום רביעי?", answer="יום חמישי."),
    # -- Reasoning without a tool, in Hebrew ----------------------------------
    InstructExample(
        user="אם היום יום רביעי, איזה יום יהיה בעוד יומיים?",
        think="יום רביעי ועוד יום זה יום חמישי, ועוד יום זה יום שישי.",
        answer="יהיה יום שישי.",
    ),
    InstructExample(
        user="יש לי שלושה תפוחים ואני אוכל אחד. אחר כך אני קונה שניים. כמה יש לי?",
        think="שלוש פחות אחד זה שתיים, ועוד שניים זה ארבע.",
        answer="יש לך ארבעה תפוחים.",
    ),
    InstructExample(
        user="מה כבד יותר, קילו ברזל או קילו נוצות?",
        think="שניהם קילו אחד; ההבדל הוא בנפח, לא במשקל.",
        answer="שניהם שווים במשקל.",
    ),
    # -- Honest ignorance, in Hebrew ------------------------------------------
    InstructExample(
        user="מי ניצח במונדיאל 2022?",
        answer="אני לא יודע. אני מודל קטן ולא אומנתי על אירועים עדכניים.",
    ),
    InstructExample(
        user="מה מזג האוויר היום?",
        answer="אני לא יכול לדעת. אין לי גישה למידע על מזג אוויר.",
    ),
    InstructExample(
        user="מה השם שלי?",
        answer="אני לא יודע את השם שלך. לא אמרת לי אותו.",
    ),
    InstructExample(
        user="מה יקרה מחר?",
        answer="אין לי דרך לדעת את זה.",
    ),
    InstructExample(
        user="כמה אנשים גרים בטוקיו?",
        answer="אני לא בטוח במספר הזה, אז מעדיף לא לנחש.",
    ),
)


#: Every hand-written example, in one tuple.
INSTRUCT_EXAMPLES: tuple[InstructExample, ...] = (
    *_CALCULATOR,
    *_CALCULATOR_OPERAND_COPYING,
    *_CALCULATOR_WITH_THINKING,
    *_CLOCK,
    *_NO_TOOL,
    *_THINKING_NO_TOOL,
    *_UNKNOWN,
    *_REBALANCE,
    *_MULTI_TURN,
    *_HEBREW,
)


def build_examples(
    examples: tuple[InstructExample, ...] = INSTRUCT_EXAMPLES,
) -> list[list[Message]]:
    """Turn the hand-written examples into message lists, running real tools.

    Every non-pinned tool result is obtained by **actually executing the tool**,
    so no tool output in the training data is invented. If a declared call does
    not run, that is a bug in the example and it raises rather than silently
    training the model on a fabricated result.
    """
    from ..tools.registry import default_registry

    registry = default_registry()
    conversations: list[list[Message]] = []

    for example in examples:
        messages: list[Message] = []
        for past_user, past_assistant in example.history:
            messages.append(user(past_user))
            messages.append(assistant(past_assistant))

        messages.append(user(example.user))

        if example.call is None:
            messages.append(assistant(example.answer, thinking=example.think))
        else:
            name, arguments = example.call
            call = ToolCall(name=name, arguments=arguments)
            messages.append(assistant("", thinking=example.think, tool_calls=[call]))

            if example.pinned:
                if example.result is None:
                    raise ValueError(f"{example.user!r}: pinned example has no result")
                result_text = example.result
            else:
                outcome = registry.execute(call)
                if outcome.is_error:
                    raise ValueError(
                        f"{example.user!r}: declared tool call failed: {outcome.content}"
                    )
                result_text = outcome.content

            messages.append(
                Message(role=Role.TOOL, content=result_text, name=name, tool_call_id=call.id)
            )
            messages.append(assistant(example.answer))

        conversations.append(messages)

    return conversations
