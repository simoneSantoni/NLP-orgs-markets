            ____                                                       _            ___
           / __ \___  ____ ____  _  __   ____ ______   ____ _   ____  (_)___  ___  / (_)___  ___
          / /_/ / _ \/ __ `/ _ \| |/_/  / __ `/ ___/  / __ `/  / __ \/ / __ \/ _ \/ / / __ \/ _ \
         / _, _/  __/ /_/ /  __/>  <   / /_/ (__  )  / /_/ /  / /_/ / / /_/ /  __/ / / / / /  __/
        /_/ |_|\___/\__, /\___/_/|_|   \__,_/____/   \__,_/  / .___/_/ .___/\___/_/_/_/ /_/\___/
                   /____/                                   /_/     /_/
        • formally, a regular expression is an "algebraic notation for characterizing a
        • set of strings" (Jurafski and Martin, 2008)
        • a key features of regex as and algebraic notation is the possibility to
        • express fairly complex strings in relatively simple terms
        • regex can also be thought as a pipeline that connect the world of natural
        • language (the one we, humans, speak) with the world of formal language (that
        • is, the language f machines)

            ____                          _          ____        __  __
           / __ \___  ____ ____  _  __   (_)___     / __ \__  __/ /_/ /_  ____  ____
          / /_/ / _ \/ __ `/ _ \| |/_/  / / __ \   / /_/ / / / / __/ __ \/ __ \/ __ \
         / _, _/  __/ /_/ /  __/>  <   / / / / /  / ____/ /_/ / /_/ / / / /_/ / / / /
        /_/ |_|\___/\__, /\___/_/|_|  /_/_/ /_/  /_/    \__, /\__/_/ /_/\____/_/ /_/
                   /____/                              /____/
        • Python implements the so-called 'extended regular expressions' parser
        • caveat: if you use other programming languages (that rely on different
        • parsersa), your expressions could not be portable
        • the `re` module for Python
        • has stellar quality documentation

            __                         _                                             __              __     _                 __     _                                     __
           / /  ___  ____ __________  (_)___  ____ _   ________  ____ ____  _  __   / /_  __  __   _/_/____(_)___ ___  ____  / /__  | |   ___  _  ______ _____ ___  ____  / /__  _____
          / /  / _ \/ __ `/ ___/ __ \/ / __ \/ __ `/  / ___/ _ \/ __ `/ _ \| |/_/  / __ \/ / / /  / // ___/ / __ `__ \/ __ \/ / _ \ / /  / _ \| |/_/ __ `/ __ `__ \/ __ \/ / _ \/ ___/
         / /___  __/ /_/ / /  / / / / / / / / /_/ /  / /  /  __/ /_/ /  __/>  <   / /_/ / /_/ /  / /(__  ) / / / / / / /_/ / /  __// /  /  __/>  </ /_/ / / / / / / /_/ / /  __(__  )
        /_____\___/\__,_/_/  /_/ /_/_/_/ /_/\__, /  /_/   \___/\__, /\___/_/|_|  /_.___/\__, /  / //____/_/_/ /_/ /_/ .___/_/\___//_/   \___/_/|_|\__,_/_/ /_/ /_/ .___/_/\___/____/
                                           /____/             /____/                   /____/   |_|                /_/          /_/                             /_/
        Consider these sample sentences:

        'Apple laptops are for starbucksers'

        'Did an apple really fall on Isaac Newton's head?'

        'How do I fix my Apple MacBookPro? I spilled apple juice over the keyboard'

        'The brand-new MacBookPro13-in costs £ 1,500'

        'Each glass of apple juice takes 3 apples'

        'run ran run'

        'The CEO of alfa announced the acquisition of beta. The reaction of
        stakeholders has been positive.'

        Let's see regex in action:

        • regex are case sensitive
        • using the `^` symbol to negate a pattern
        • 'the preceding character or nothing'
        • between `a` and `c`
        • path repetition
        • one or more occurrences of the immediately preceding character
        • using the `^` symbol to match the start of a line
        • disjunctions
        • grouping

            ____                    __                                          __
           / __ \____  ____  __  __/ /___ ______   ____  ____  ___  _________ _/ /_____  __________
          / /_/ / __ \/ __ \/ / / / / __ `/ ___/  / __ \/ __ \/ _ \/ ___/ __ `/ __/ __ \/ ___/ ___/
         / ____/ /_/ / /_/ / /_/ / / /_/ / /     / /_/ / /_/ /  __/ /  / /_/ / /_/ /_/ / /  (__  )
        /_/    \____/ .___/\__,_/_/\__,_/_/      \____/ .___/\___/_/   \__,_/\__/\____/_/  /____/
                   /_/                               /_/
        | RE | Expansion    | Match              | First Matches     |
        |----|--------------|--------------------|-------------------|
        | d  | [0-9]        | any digit          | Party?of?5        |
        | \D | [ˆ0-9]       | any non-digit      | Blue?moon         |
        | \w | [a-zA-Z0-9_] | any alphanumeric   | /underscore Daiyu |
        | \W | [ˆ\w]        | a non-alphanumeric | !!!!              |
        | \s | [?\r\t\n\f]  | whitespace         | (space, tab)      |
        | \S | [ˆ\s]        | Non-whitespace     | in?Concord        |

        Source: Jurafski and Martin, 2008

         _       __               __          __            __                                                           __   __            __
        | |     / /___  _________/ /____     / /____  _  __/ /_   _________  _________  __  _______     ____ _____  ____/ /  / /____  _  __/ /_   _________  _________  ____  _________ _
        | | /| / / __ \/ ___/ __  / ___/    / __/ _ \| |/_/ __/  / ___/ __ \/ ___/ __ \/ / / / ___/    / __ `/ __ \/ __  /  / __/ _ \| |/_/ __/  / ___/ __ \/ ___/ __ \/ __ \/ ___/ __ `/
        | |/ |/ / /_/ / /  / /_/ (__  )    / /_/  __/>  </ /_   / /__/ /_/ / /  / /_/ / /_/ (__  )    / /_/ / / / / /_/ /  / /_/  __/>  </ /_   / /__/ /_/ / /  / /_/ / /_/ / /  / /_/ /
        |__/|__/\____/_/   \__,_/____( )   \__/\___/_/|_|\__/   \___/\____/_/  / .___/\__,_/____( )   \__,_/_/ /_/\__,_/   \__/\___/_/|_|\__/   \___/\____/_/  / .___/\____/_/   \__,_/
                                     |/                                       /_/               |/                                                            /_/
        • words are the fundamental unit of observations
        • words that appear in a same document constitute a corpus of text
        • collections of documents constitute text corpora

          ______          __                                      ___            __  _
         /_  __/__  _  __/ /_   ____  ____  _________ ___  ____ _/ (_)___ ____ _/ /_(_)___  ____
          / / / _ \| |/_/ __/  / __ \/ __ \/ ___/ __ `__ \/ __ `/ / /_  // __ `/ __/ / __ \/ __ \
         / / /  __/>  </ /_   / / / / /_/ / /  / / / / / / /_/ / / / / /_ /_/ / /_/ / /_/ / / / /
        /_/  \___/_/|_|\__/  /_/ /_/\____/_/  /_/ /_/ /_/\__,_/_/_/ /___\__,_/\__/_/\____/_/ /_/

        Before almost any natural language processing of a text, the text has to be
        normalized. At least three tasks are commonly applied as part of any
        normalization process:

        1. Tokenizing (segmenting) words
        2. Normalizing word formats
        3. Segmenting sentences

















































































slide 002
