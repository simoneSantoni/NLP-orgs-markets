Week 1, problem set
===================

Consider the below [displayed text][1]:

```
Thomas Wesley Pentz (born November 10, 1978),[2] known professionally as Diplo,
is an American DJ, songwriter, and record producer based in Los Angeles,
California. He is the co-creator and lead member of the electronic dancehall
music project Major Lazer, a member of the supergroup LSD with Sia and Labrinth,
a member of electronic duo Jack Ü with producer and DJ Skrillex, and a member of
Silk City with Mark Ronson. He founded and manages record company Mad Decent, as
well as co-founding the non-profit organization Heaps Decent. Among other jobs,
he has worked as an English teacher in Japan[3] and a school teacher in
Philadelphia. His 2013 EP, Revolution, debuted at number 68 on the US Billboard
200. The EP's title track was later featured in a commercial for Hyundai and is
featured on the WWE 2K16 soundtrack.

During his rise to fame, Diplo worked with British musician M.I.A., an artist
who is credited with giving him exposure in his early career. Later, he and
fellow M.I.A. producer Switch created a Jamaican dancehall project and cartoon
series titled Major Lazer.[4] Since then, Diplo has worked on production and
mixtape projects with many other pop artists, such as Gwen Stefani, Die
Antwoord, Britney Spears, Madonna, Shakira, Beyoncé, Ellie Goulding, No Doubt,
Justin Bieber, Usher, Snoop Dogg, Trippie Redd, Chris Brown, CL, G-Dragon, Bad
Bunny, MØ and Poppy.[5][6][7][8][9] His alias, short for Diplodocus, derives
from his childhood fascination with dinosaurs.[10]
```

Using pure Python code and the `re` module:

1. extract the birth date of Diplo (note: stick with the format reported in the
    raw data, namely, `month` (in letters) + `whitespace` + `day` + `comma` +
    `year`
2. extract the number at which Diplo's album Revolution debuted in 2013
3. extract the names of Diplo's collaborators


    [1]: https://en.wikipedia.org/wiki/Diplo
