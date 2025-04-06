id2pair: dict[int: int | str] = {0: '\n', 1: ' ', 2: '!', 3: '(', 4: ')', 5: '*', 6: ',', 7: '-', 8: '.', 9: '0', 10: '1', 11: '2', 12: '3', 13: '4', 14: '5', 15: '6', 16: '7', 17: '8', 18: '9', 19: ':', 20: ';', 21: '?', 22: 'I', 23: 'X', 24: 'e', 25: 'f', 26: 'o', 27: '|', 28: '\xa0', 29: '«', 30: '»', 31: 'А', 32: 'Б', 33: 'В', 34: 'Г', 35: 'Д', 36: 'Е', 37: 'Ж', 38: 'З', 39: 'И', 40: 'К', 41: 'Л', 42: 'М', 43: 'Н', 44: 'О', 45: 'П', 46: 'Р', 47: 'С', 48: 'Т', 49: 'У', 50: 'Ф', 51: 'Х', 52: 'Ц', 53: 'Ч', 54: 'Ш', 55: 'Э', 56: 'Ю', 57: 'Я', 58: 'а', 59: 'б', 60: 'в', 61: 'г', 62: 'д', 63: 'е', 64: 'ж', 65: 'з', 66: 'и', 67: 'й', 68: 'к', 69: 'л', 70: 'м', 71: 'н', 72: 'о', 73: 'п', 74: 'р', 75: 'с', 76: 'т', 77: 'у', 78: 'ф', 79: 'х', 80: 'ц', 81: 'ч', 82: 'ш', 83: 'щ', 84: 'ъ', 85: 'ы', 86: 'ь', 87: 'э', 88: 'ю', 89: 'я', 90: 'ё', 91: '–', 92: '—', 93: '…', 94: (6, 1), 95: (72, 1), 96: (75, 76), 97: (66, 1), 98: (63, 1), 99: (58, 1), 100: (89, 1), 101: (63, 71), 102: (74, 58), 103: (8, 1), 104: (71, 72), 105: (86, 1), 106: (74, 72), 107: (73, 72), 108: (66, 76), 109: (60, 72), 110: (71, 58), 111: (76, 72), 112: (63, 69), 113: (67, 1), 114: (68, 72), 115: (63, 76), 116: (70, 1), 117: (68, 58), 118: (71, 66), 119: (60, 58), 120: (60, 1), 121: (71, 85), 122: (63, 74), 123: (61, 95), 124: (75, 100), 125: (68, 1), 126: (101, 66), 127: (58, 76), 128: (72, 76), 129: (71, 63), 130: (74, 63), 131: (66, 65), 132: (79, 1), 133: (77, 1), 134: (69, 86), 135: (76, 95), 136: (71, 98), 137: (75, 72), 138: (73, 106), 139: (71, 95), 140: (74, 66), 141: (62, 72), 142: (71, 99), 143: (60, 63), 144: (85, 1), 145: (61, 72), 146: (70, 72), 147: (68, 66), 148: (66, 75), 149: (96, 72), 150: (69, 72), 151: (74, 77), 152: (62, 58), 153: (65, 58), 154: (115, 1), 155: (69, 1), 156: (69, 63), 157: (60, 66), 158: (88, 1), 159: (60, 75), 160: (71, 1), 161: (76, 1), 162: (81, 63), 163: (112, 86), 164: (73, 140), 165: (96, 60), 166: (69, 58), 167: (8, 0), 168: (70, 66), 169: (76, 66), 170: (104, 96), 171: (62, 77), 172: (81, 58), 173: (96, 58), 174: (69, 88), 175: (59, 85), 176: (76, 63), 177: (71, 77), 178: (75, 109), 179: (72, 59), 180: (71, 97), 181: (68, 77), 182: (75, 58), 183: (81, 135), 184: (69, 66), 185: (64, 66), 186: (64, 131), 187: (74, 108), 188: (111, 74), 189: (89, 94), 190: (102, 65), 191: (68, 95), 192: (62, 63), 193: (75, 75), 194: (126, 98), 195: (112, 72), 196: (76, 58), 197: (67, 94), 198: (63, 123), 199: (70, 94), 200: (86, 94), 201: (70, 133), 202: (69, 97), 203: (63, 94), 204: (75, 1), 205: (115, 124), 206: (76, 105), 207: (70, 63), 208: (92, 1), 209: (70, 58), 210: (195, 143), 211: (127, 112), 212: (70, 101), 213: (65, 110), 214: (81, 210), 215: (76, 77), 216: (59, 72), 217: (72, 160), 218: (66, 94), 219: (60, 85), 220: (103, 33), 221: (187, 66), 222: (81, 66), 223: (75, 114), 224: (70, 77), 225: (62, 151), 226: (73, 148), 227: (19, 1), 228: (81, 108), 229: (59, 63), 230: (83, 66), 231: (61, 58), 232: (79, 72), 233: (108, 98), 234: (130, 62), 235: (61, 62), 236: (73, 122), 237: (68, 99), 238: (69, 100), 239: (85, 119), 240: (117, 125), 241: (63, 116), 242: (96, 66), 243: (60, 69), 244: (64, 98), 245: (68, 221), 246: (64, 62), 247: (103, 45), 248: (58, 60), 249: (75, 63), 250: (62, 66), 251: (86, 124), 252: (96, 102), 253: (63, 113), 254: (63, 96), 255: (82, 66), 256: (88, 161), 257: (48, 72), 258: (117, 64), 259: (143, 62), 260: (121, 113), 261: (89, 76), 262: (87, 111), 263: (134, 191), 264: (76, 109), 265: (109, 75), 266: (107, 62), 267: (70, 71), 268: (107, 69), 269: (94, 183), 270: (137, 59), 271: (146, 64), 272: (64, 58), 273: (121, 132), 274: (170, 105), 275: (264, 74), 276: (75, 60), 277: (215, 74), 278: (94, 97), 279: (81, 77), 280: (70, 97), 281: (69, 99), 282: (90, 1), 283: (117, 65), 284: (126, 100), 285: (257, 69), 286: (285, 149), 287: (66, 28), 288: (77, 82), 289: (248, 188), 290: (99, 97), 291: (104, 60), 292: (103, 44), 293: (75, 77), 294: (121, 1), 295: (249, 59), 296: (103, 35), 297: (127, 105), 298: (70, 85), 299: (103, 43), 300: (275, 162), 301: (68, 96), 302: (62, 238), 303: (104, 123), 304: (73, 58), 305: (69, 95), 306: (75, 66), 307: (72, 149), 308: (75, 150), 309: (104, 113), 310: (114, 113), 311: (60, 122), 312: (226, 211), 313: (87, 135), 314: (165, 101), 315: (104, 94), 316: (182, 146), 317: (88, 94), 318: (103, 39), 319: (75, 156), 320: (69, 148), 321: (30, 94), 322: (61, 166), 323: (185, 109), 324: (59, 128), 325: (174, 62), 326: (128, 104), 327: (307, 63), 328: (327, 159), 329: (171, 82), 330: (83, 58), 331: (94, 114), 332: (75, 147), 333: (61, 66), 334: (68, 97), 335: (61, 1), 336: (229, 65), 337: (111, 263), 338: (121, 98), 339: (243, 89), 340: (108, 105), 341: (76, 124), 342: (235, 99), 343: (172, 96), 344: (288, 147), 345: (60, 77), 346: (157, 62), 347: (62, 95), 348: (59, 144), 349: (71, 89), 350: (73, 77), 351: (68, 94), 352: (188, 85), 353: (72, 62), 354: (69, 124), 355: (186, 71), 356: (70, 122), 357: (320, 105), 358: (108, 63), 359: (127, 163), 360: (108, 163), 361: (110, 100), 362: (102, 324), 363: (72, 173), 364: (165, 99), 365: (276, 89), 366: (8, 27), 367: (366, 24), 368: (367, 26), 369: (368, 25), 370: (369, 27), 371: (370, 0), 372: (1, 97), 373: (102, 60), 374: (103, 40), 375: (236, 63), 376: (87, 76), 377: (80, 66), 378: (96, 109), 379: (76, 151), 380: (141, 69), 381: (80, 101), 382: (254, 105), 383: (358, 102), 384: (75, 107), 385: (228, 211), 386: (138, 169), 387: (8, 8), 388: (73, 66), 389: (102, 193), 390: (60, 130), 391: (60, 99), 392: (89, 75), 393: (159, 63), 394: (145, 94), 395: (82, 58), 396: (186, 118), 397: (62, 1), 398: (265, 164), 399: (66, 132), 400: (70, 144), 401: (73, 63), 402: (186, 180), 403: (242, 79), 404: (138, 131), 405: (102, 75), 406: (75, 189), 407: (69, 383), 408: (85, 94), 409: (62, 108), 410: (61, 63), 411: (63, 67), 412: (72, 180), 413: (79, 94), 414: (99, 120), 415: (176, 301), 416: (90, 116), 417: (104, 116), 418: (63, 70), 419: (62, 112), 420: (73, 234), 421: (60, 94), 422: (114, 123), 423: (170, 97), 424: (131, 1), 425: (295, 98), 426: (179, 190), 427: (111, 102), 428: (196, 125), 429: (97, 107), 430: (128, 1), 431: (75, 89), 432: (220, 1), 433: (103, 55), 434: (121, 116), 435: (108, 251), 436: (59, 77), 437: (271, 154), 438: (326, 82), 439: (323, 76), 440: (258, 233), 441: (267, 194), 442: (8, 43), 443: (181, 134), 444: (443, 277), 445: (72, 73), 446: (149, 89), 447: (88, 230), 448: (90, 74), 449: (65, 66), 450: (60, 95), 451: (146, 113), 452: (60, 102), 453: (104, 98), 454: (175, 305), 455: (148, 181), 456: (75, 122), 457: (283, 239), 458: (61, 77), 459: (157, 81), 460: (68, 174), 461: (96, 119), 462: (150, 64), 463: (68, 74), 464: (81, 101), 465: (30, 1), 466: (69, 77), 467: (35, 328), 468: (106, 69), 469: (60, 28), 470: (72, 75), 471: (258, 141), 472: (108, 124), 473: (185, 119), 474: (75, 166), 475: (59, 156), 476: (119, 155), 477: (114, 352), 478: (90, 71), 479: (60, 115), 480: (293, 83), 481: (455, 75), 482: (59, 66), 483: (164, 356), 484: (168, 74), 485: (404, 259), 486: (106, 62), 487: (407, 277), 488: (145, 62), 489: (110, 82), 490: (88, 341), 491: (459, 1), 492: (48, 58), 493: (111, 81), 494: (80, 63), 495: (75, 105), 496: (101, 139), 497: (119, 64), 498: (122, 63), 499: (439, 121), 500: (89, 339), 501: (410, 106), 502: (43, 58), 503: (388, 82), 504: (247, 72), 505: (212, 76), 506: (75, 70), 507: (69, 85), 508: (118, 132), 509: (235, 98), 510: (112, 1), 511: (76, 74), 512: (65, 74), 513: (70, 104), 514: (73, 130), 515: (279, 60), 516: (129, 107), 517: (193, 66), 518: (76, 86), 519: (59, 69), 520: (192, 67), 521: (75, 234), 522: (318, 1), 523: (379, 62), 524: (232, 150), 525: (332, 113), 526: (151, 193), 527: (109, 65), 528: (76, 200), 529: (70, 127), 530: (380, 64), 531: (106, 60), 532: (343, 86), 533: (384, 270), 534: (433, 135), 535: (137, 65), 536: (89, 70), 537: (69, 126), 538: (75, 435), 539: (214, 237), 540: (77, 308), 541: (540, 157), 542: (62, 122), 543: (70, 100), 544: (152, 244), 545: (355, 105), 546: (436, 62), 547: (177, 158), 548: (118, 209), 549: (386, 109), 550: (286, 123), 551: (71, 86), 552: (300, 364), 553: (31, 8), 554: (47, 247), 555: (554, 344), 556: (102, 61), 557: (98, 444), 558: (144, 107), 559: (58, 94), 560: (73, 85), 561: (292, 160), 562: (61, 102), 563: (252, 71), 564: (162, 123), 565: (376, 97), 566: (63, 83), 567: (566, 282), 568: (398, 89), 569: (568, 169), 570: (76, 144), 571: (115, 66), 572: (225, 145), 573: (175, 202), 574: (111, 106), 575: (65, 99), 576: (97, 60), 577: (262, 123), 578: (76, 122), 579: (100, 136), 580: (168, 102), 581: (73, 74), 582: (175, 206), 583: (156, 68), 584: (137, 322), 585: (112, 66), 586: (73, 95), 587: (114, 98), 588: (141, 106), 589: (103, 31), 590: (147, 113), 591: (90, 161), 592: (72, 270), 593: (126, 158), 594: (145, 116), 595: (173, 291), 596: (319, 171), 597: (535, 152), 598: (117, 94), 599: (72, 140), 600: (78, 72), 601: (227, 29), 602: (524, 62), 603: (325, 97), 604: (360, 139), 605: (71, 76), 606: (225, 335), 607: (262, 116), 608: (8, 42), 609: (96, 130), 610: (426, 72), 611: (68, 151), 612: (225, 231), 613: (61, 110), 614: (245, 147), 615: (245, 237), 616: (28, 208), 617: (71, 126), 618: (390, 212), 619: (58, 134), 620: (553, 555), 621: (62, 391), 622: (290, 77), 623: (207, 246), 624: (623, 133), 625: (470, 291), 626: (117, 153), 627: (69, 90), 628: (252, 152), 629: (118, 100), 630: (427, 100), 631: (216, 134), 632: (89, 132), 633: (109, 69), 634: (159, 282), 635: (512, 284), 636: (515, 378), 637: (94, 107), 638: (58, 74), 639: (96, 105), 640: (296, 238), 641: (519, 131), 642: (167, 45), 643: (107, 87), 644: (480, 254), 645: (178, 158), 646: (355, 86), 647: (72, 142), 648: (136, 107), 649: (178, 253), 650: (69, 94), 651: (178, 97), 652: (81, 127), 653: (7, 135), 654: (63, 65), 655: (228, 163), 656: (88, 83), 657: (457, 154), 658: (103, 42), 659: (74, 89), 660: (477, 98), 661: (81, 122), 662: (68, 405), 663: (232, 409), 664: (65, 72), 665: (75, 532), 666: (182, 63), 667: (153, 460), 668: (138, 475), 669: (167, 492), 670: (179, 102), 671: (190, 298), 672: (671, 82), 673: (167, 39), 674: (595, 472), 675: (100, 107), 676: (449, 377), 677: (163, 65), 678: (677, 579), 679: (66, 242), 680: (87, 301), 681: (680, 130), 682: (681, 209), 683: (682, 134), 684: (76, 97), 685: (76, 89), 686: (685, 64), 687: (165, 95), 688: (75, 202), 689: (94, 99), 690: (222, 110), 691: (61, 466), 692: (75, 207), 693: (82, 63), 694: (331, 342), 695: (107, 462), 696: (88, 76), 697: (82, 105), 698: (66, 202), 699: (73, 102), 700: (216, 156), 701: (111, 201), 702: (463, 239), 703: (97, 110), 704: (506, 128), 705: (116, 610), 706: (438, 194), 707: (467, 114), 708: (28, 1), 709: (708, 0), 710: (77, 76), 711: (119, 206), 712: (63, 250), 713: (214, 125), 714: (101, 1), 715: (296, 328), 716: (442, 103), 717: (500, 205), 718: (125, 499), 719: (613, 111), 720: (268, 85), 721: (720, 551), 722: (77, 111), 723: (536, 230), 724: (468, 105), 725: (502, 503), 726: (725, 233), 727: (726, 137), 728: (727, 222), 729: (728, 617), 730: (729, 63), 731: (730, 7), 732: (731, 389), 733: (732, 77), 734: (733, 246), 735: (734, 194), 736: (735, 142), 737: (736, 176), 738: (737, 201), 739: (504, 440), 740: (739, 441), 741: (740, 289), 742: (741, 414), 743: (742, 178), 744: (743, 241), 745: (744, 415), 746: (745, 203), 747: (746, 164), 748: (747, 259), 749: (748, 233), 750: (749, 621), 751: (750, 78), 752: (751, 556), 753: (752, 505), 754: (753, 622), 755: (754, 440), 756: (755, 365), 757: (756, 65), 758: (757, 105), 759: (758, 624), 760: (759, 118), 761: (760, 168), 762: (99, 625), 763: (762, 557), 764: (763, 558), 765: (764, 392), 766: (765, 71), 767: (766, 233), 768: (767, 178), 769: (768, 98), 770: (769, 441), 771: (770, 125), 772: (771, 289), 773: (772, 223), 774: (773, 224), 775: (774, 371), 776: (174, 216), 777: (114, 120), 778: (136, 213), 779: (118, 80), 780: (121, 197), 781: (65, 418), 782: (236, 105), 783: (103, 48), 784: (129, 75), 785: (170, 86), 786: (129, 141), 787: (332, 132), 788: (71, 261), 789: (168, 106), 790: (516, 521), 791: (790, 314), 792: (79, 58), 793: (20, 1), 794: (110, 191), 795: (99, 136), 796: (179, 83), 797: (295, 100), 798: (115, 94), 799: (77, 70), 800: (96, 106), 801: (69, 406), 802: (471, 123), 803: (103, 47), 804: (243, 194), 805: (90, 341), 806: (64, 112), 807: (21, 1), 808: (212, 139), 809: (91, 1), 810: (200, 183), 811: (331, 352), 812: (365, 153), 813: (89, 154), 814: (223, 123), 815: (141, 482), 816: (60, 340), 817: (111, 594), 818: (817, 672), 819: (818, 537), 820: (819, 113), 821: (675, 676), 822: (821, 89), 823: (822, 227), 824: (180, 120), 825: (678, 584), 826: (825, 538), 827: (826, 204), 828: (827, 493), 829: (828, 310), 830: (829, 635), 831: (248, 427), 832: (261, 124), 833: (483, 94), 834: (115, 269), 835: (81, 194), 836: (214, 598), 837: (48, 122), 838: (837, 187), 839: (838, 599), 840: (118, 117), 841: (111, 116), 842: (268, 77), 843: (175, 281), 844: (311, 177), 845: (177, 64), 846: (177, 76), 847: (76, 90), 848: (322, 60), 849: (299, 95), 850: (63, 227), 851: (109, 138), 852: (101, 144), 853: (333, 98), 854: (312, 105), 855: (149, 106), 856: (87, 169), 857: (65, 1), 858: (609, 70), 859: (706, 125), 860: (62, 345), 861: (860, 132), 862: (64, 118), 863: (387, 103), 864: (311, 82), 865: (59, 106), 866: (126, 89), 867: (85, 98), 868: (271, 139), 869: (498, 666), 870: (226, 359), 871: (716, 550), 872: (50, 608), 873: (872, 715), 874: (108, 1), 875: (77, 158), 876: (166, 495), 877: (261, 163), 878: (292, 62), 879: (169, 80), 880: (136, 146), 881: (81, 105), 882: (719, 491), 883: (138, 494), 884: (883, 193), 885: (129, 28), 886: (274, 97), 887: (58, 68), 888: (32, 72), 889: (159, 90), 890: (374, 58), 891: (110, 446), 892: (891, 230), 893: (62, 133), 894: (129, 123), 895: (167, 44), 896: (176, 782), 897: (447, 113), 898: (331, 630), 899: (110, 280), 900: (631, 82), 901: (72, 71), 902: (162, 787), 903: (349, 155), 904: (97, 97), 905: (121, 94), 906: (66, 192), 907: (74, 85), 908: (101, 86), 909: (224, 94), 910: (140, 80), 911: (71, 100), 912: (197, 97), 913: (114, 574), 914: (109, 250), 915: (128, 66), 916: (62, 297), 917: (167, 40), 918: (111, 394), 919: (116, 72), 920: (136, 337), 921: (68, 118), 922: (403, 97), 923: (117, 68), 924: (923, 653), 925: (319, 62), 926: (357, 120), 927: (114, 129), 928: (94, 217), 929: (107, 168), 930: (929, 110), 931: (81, 448), 932: (147, 119), 933: (68, 203), 934: (64, 203), 935: (65, 89), 936: (585, 114), 937: (109, 134), 938: (316, 98), 939: (153, 70), 940: (110, 116), 941: (336, 1), 942: (181, 62), 943: (60, 188), 944: (59, 166), 945: (944, 145), 946: (63, 199), 947: (168, 94), 948: (66, 149), 949: (340, 97), 950: (662, 128), 951: (81, 86), 952: (174, 59), 953: (393, 342), 954: (60, 184), 955: (3, 45), 956: (67, 79), 957: (956, 101), 958: (957, 119), 959: (958, 134), 960: (5, 4), 961: (3, 10), 962: (961, 17), 963: (7, 10), 964: (963, 18), 965: (1, 208), 966: (667, 172), 967: (966, 205), 968: (592, 101), 969: (110, 397), 970: (420, 462), 971: (970, 101), 972: (620, 142), 973: (463, 85), 974: (223, 263), 975: (114, 342), 976: (289, 99), 977: (159, 98), 978: (330, 205), 979: (812, 294), 980: (153, 66), 981: (116, 110), 982: (485, 284), 983: (673, 820), 984: (674, 596), 985: (984, 88), 986: (985, 330), 987: (986, 823), 988: (72, 139), 989: (167, 43), 990: (989, 830), 991: (831, 296), 992: (991, 411), 993: (992, 165), 994: (993, 360), 995: (994, 315), 996: (110, 833), 997: (106, 209), 998: (128, 184), 999: (121, 280), 1000: (373, 104), 1001: (1000, 329), 1002: (152, 74), 1003: (761, 299), 1004: (1003, 775), 1005: (77, 65), 1006: (72, 464), 1007: (1006, 105), 1008: (541, 89), 1009: (640, 577), 1010: (1009, 107), 1011: (1010, 110), 1012: (1011, 815), 1013: (68, 135), 1014: (103, 53), 1015: (839, 218), 1016: (65, 127), 1017: (529, 122), 1018: (1017, 66), 1019: (299, 98), 1020: (66, 543), 1021: (390, 543), 1022: (102, 62), 1023: (1022, 97)}
id2text: dict[int: str] = {0: '\n', 1: ' ', 2: '!', 3: '(', 4: ')', 5: '*', 6: ',', 7: '-', 8: '.', 9: '0', 10: '1', 11: '2', 12: '3', 13: '4', 14: '5', 15: '6', 16: '7', 17: '8', 18: '9', 19: ':', 20: ';', 21: '?', 22: 'I', 23: 'X', 24: 'e', 25: 'f', 26: 'o', 27: '|', 28: '\xa0', 29: '«', 30: '»', 31: 'А', 32: 'Б', 33: 'В', 34: 'Г', 35: 'Д', 36: 'Е', 37: 'Ж', 38: 'З', 39: 'И', 40: 'К', 41: 'Л', 42: 'М', 43: 'Н', 44: 'О', 45: 'П', 46: 'Р', 47: 'С', 48: 'Т', 49: 'У', 50: 'Ф', 51: 'Х', 52: 'Ц', 53: 'Ч', 54: 'Ш', 55: 'Э', 56: 'Ю', 57: 'Я', 58: 'а', 59: 'б', 60: 'в', 61: 'г', 62: 'д', 63: 'е', 64: 'ж', 65: 'з', 66: 'и', 67: 'й', 68: 'к', 69: 'л', 70: 'м', 71: 'н', 72: 'о', 73: 'п', 74: 'р', 75: 'с', 76: 'т', 77: 'у', 78: 'ф', 79: 'х', 80: 'ц', 81: 'ч', 82: 'ш', 83: 'щ', 84: 'ъ', 85: 'ы', 86: 'ь', 87: 'э', 88: 'ю', 89: 'я', 90: 'ё', 91: '–', 92: '—', 93: '…', 94: ', ', 95: 'о ', 96: 'ст', 97: 'и ', 98: 'е ', 99: 'а ', 100: 'я ', 101: 'ен', 102: 'ра', 103: '. ', 104: 'но', 105: 'ь ', 106: 'ро', 107: 'по', 108: 'ит', 109: 'во', 110: 'на', 111: 'то', 112: 'ел', 113: 'й ', 114: 'ко', 115: 'ет', 116: 'м ', 117: 'ка', 118: 'ни', 119: 'ва', 120: 'в ', 121: 'ны', 122: 'ер', 123: 'го ', 124: 'ся ', 125: 'к ', 126: 'ени', 127: 'ат', 128: 'от', 129: 'не', 130: 'ре', 131: 'из', 132: 'х ', 133: 'у ', 134: 'ль', 135: 'то ', 136: 'не ', 137: 'со', 138: 'про', 139: 'но ', 140: 'ри', 141: 'до', 142: 'на ', 143: 'ве', 144: 'ы ', 145: 'го', 146: 'мо', 147: 'ки', 148: 'ис', 149: 'сто', 150: 'ло', 151: 'ру', 152: 'да', 153: 'за', 154: 'ет ', 155: 'л ', 156: 'ле', 157: 'ви', 158: 'ю ', 159: 'вс', 160: 'н ', 161: 'т ', 162: 'че', 163: 'ель', 164: 'при', 165: 'ств', 166: 'ла', 167: '.\n', 168: 'ми', 169: 'ти', 170: 'ност', 171: 'ду', 172: 'ча', 173: 'ста', 174: 'лю', 175: 'бы', 176: 'те', 177: 'ну', 178: 'сво', 179: 'об', 180: 'ни ', 181: 'ку', 182: 'са', 183: 'что ', 184: 'ли', 185: 'жи', 186: 'жиз', 187: 'рит', 188: 'тор', 189: 'я, ', 190: 'раз', 191: 'ко ', 192: 'де', 193: 'сс', 194: 'ение ', 195: 'ело', 196: 'та', 197: 'й, ', 198: 'его ', 199: 'м, ', 200: 'ь, ', 201: 'му ', 202: 'ли ', 203: 'е, ', 204: 'с ', 205: 'ется ', 206: 'ть ', 207: 'ме', 208: '— ', 209: 'ма', 210: 'елове', 211: 'ател', 212: 'мен', 213: 'зна', 214: 'челове', 215: 'ту', 216: 'бо', 217: 'он ', 218: 'и, ', 219: 'вы', 220: '. В', 221: 'рити', 222: 'чи', 223: 'ско', 224: 'му', 225: 'дру', 226: 'пис', 227: ': ', 228: 'чит', 229: 'бе', 230: 'щи', 231: 'га', 232: 'хо', 233: 'ите ', 234: 'ред', 235: 'гд', 236: 'пер', 237: 'ка ', 238: 'ля ', 239: 'ыва', 240: 'как ', 241: 'ем ', 242: 'сти', 243: 'вл', 244: 'же ', 245: 'крити', 246: 'жд', 247: '. П', 248: 'ав', 249: 'се', 250: 'ди', 251: 'ься ', 252: 'стра', 253: 'ей ', 254: 'ест', 255: 'ши', 256: 'ют ', 257: 'То', 258: 'каж', 259: 'вед', 260: 'ный ', 261: 'ят', 262: 'это', 263: 'лько ', 264: 'тво', 265: 'вос', 266: 'под', 267: 'мн', 268: 'пол', 269: ', что ', 270: 'соб', 271: 'мож', 272: 'жа', 273: 'ных ', 274: 'ность ', 275: 'твор', 276: 'св', 277: 'тур', 278: ', и ', 279: 'чу', 280: 'ми ', 281: 'ла ', 282: 'ё ', 283: 'каз', 284: 'ения ', 285: 'Тол', 286: 'Толсто', 287: 'и\xa0', 288: 'уш', 289: 'автор', 290: 'а и ', 291: 'нов', 292: '. О', 293: 'су', 294: 'ны ', 295: 'себ', 296: '. Д', 297: 'ать ', 298: 'мы', 299: '. Н', 300: 'творче', 301: 'кст', 302: 'для ', 303: 'ного ', 304: 'па', 305: 'ло ', 306: 'си', 307: 'осто', 308: 'сло', 309: 'ной ', 310: 'кой ', 311: 'вер', 312: 'писател', 313: 'это ', 314: 'ствен', 315: 'но, ', 316: 'само', 317: 'ю, ', 318: '. И', 319: 'сле', 320: 'лис', 321: '», ', 322: 'гла', 323: 'живо', 324: 'бот', 325: 'люд', 326: 'отно', 327: 'остое', 328: 'остоевс', 329: 'душ', 330: 'ща', 331: ', ко', 332: 'ски', 333: 'ги', 334: 'ки ', 335: 'г ', 336: 'без', 337: 'только ', 338: 'ные ', 339: 'вля', 340: 'ить ', 341: 'тся ', 342: 'гда ', 343: 'част', 344: 'ушки', 345: 'ву', 346: 'вид', 347: 'до ', 348: 'бы ', 349: 'ня', 350: 'пу', 351: 'к, ', 352: 'торы', 353: 'од', 354: 'лся ', 355: 'жизн', 356: 'мер', 357: 'лись ', 358: 'ите', 359: 'атель', 360: 'итель', 361: 'ная ', 362: 'работ', 363: 'оста', 364: 'ства ', 365: 'свя', 366: '.|', 367: '.|e', 368: '.|eo', 369: '.|eof', 370: '.|eof|', 371: '.|eof|\n', 372: ' и ', 373: 'рав', 374: '. К', 375: 'пере', 376: 'эт', 377: 'ци', 378: 'ство', 379: 'тру', 380: 'дол', 381: 'цен', 382: 'есть ', 383: 'итера', 384: 'спо', 385: 'читател', 386: 'проти', 387: '..', 388: 'пи', 389: 'расс', 390: 'вре', 391: 'ва ', 392: 'яс', 393: 'все', 394: 'го, ', 395: 'ша', 396: 'жизни', 397: 'д ', 398: 'воспри', 399: 'их ', 400: 'мы ', 401: 'пе', 402: 'жизни ', 403: 'стих', 404: 'произ', 405: 'рас', 406: 'ся, ', 407: 'литера', 408: 'ы, ', 409: 'дит', 410: 'ге', 411: 'ей', 412: 'они ', 413: 'х, ', 414: 'а в ', 415: 'текст', 416: 'ём ', 417: 'ном ', 418: 'ем', 419: 'дел', 420: 'пред', 421: 'в, ', 422: 'кого ', 423: 'ности ', 424: 'из ', 425: 'себе ', 426: 'образ', 427: 'тора', 428: 'так ', 429: 'и по', 430: 'от ', 431: 'ся', 432: '. В ', 433: '. Э', 434: 'ным ', 435: 'иться ', 436: 'бу', 437: 'может ', 438: 'отнош', 439: 'живот', 440: 'кажите ', 441: 'мнение ', 442: '.Н', 443: 'куль', 444: 'культур', 445: 'оп', 446: 'стоя', 447: 'ющи', 448: 'ёр', 449: 'зи', 450: 'во ', 451: 'мой ', 452: 'вра', 453: 'ное ', 454: 'было ', 455: 'иску', 456: 'сер', 457: 'казыва', 458: 'гу', 459: 'вич', 460: 'клю', 461: 'ства', 462: 'лож', 463: 'кр', 464: 'чен', 465: '» ', 466: 'лу', 467: 'Достоевс', 468: 'рол', 469: 'в\xa0', 470: 'ос', 471: 'каждо', 472: 'ится ', 473: 'жива', 474: 'сла', 475: 'бле', 476: 'вал ', 477: 'которы', 478: 'ён', 479: 'вет', 480: 'сущ', 481: 'искус', 482: 'би', 483: 'пример', 484: 'мир', 485: 'произвед', 486: 'род', 487: 'литератур', 488: 'год', 489: 'наш', 490: 'ются ', 491: 'вич ', 492: 'Та', 493: 'точ', 494: 'це', 495: 'сь ', 496: 'енно ', 497: 'важ', 498: 'ере', 499: 'животны', 500: 'явля', 501: 'геро', 502: 'На', 503: 'пиш', 504: '. По', 505: 'мент', 506: 'см', 507: 'лы', 508: 'них ', 509: 'где ', 510: 'ел ', 511: 'тр', 512: 'зр', 513: 'мно', 514: 'пре', 515: 'чув', 516: 'непо', 517: 'сси', 518: 'ть', 519: 'бл', 520: 'дей', 521: 'сред', 522: '. И ', 523: 'труд', 524: 'холо', 525: 'ский ', 526: 'русс', 527: 'воз', 528: 'ть, ', 529: 'мат', 530: 'долж', 531: 'ров', 532: 'часть', 533: 'способ', 534: '. Это ', 535: 'соз', 536: 'ям', 537: 'лени', 538: 'ситься ', 539: 'человека ', 540: 'усло', 541: 'услови', 542: 'дер', 543: 'мя ', 544: 'даже ', 545: 'жизнь ', 546: 'буд', 547: 'ную ', 548: 'нима', 549: 'противо', 550: 'Толстого ', 551: 'нь', 552: 'творчества ', 553: 'А.', 554: 'С. П', 555: 'С. Пушки', 556: 'раг', 557: 'е культур', 558: 'ы по', 559: 'а, ', 560: 'пы', 561: '. Он ', 562: 'гра', 563: 'стран', 564: 'чего ', 565: 'эти ', 566: 'ещ', 567: 'ещё ', 568: 'восприя', 569: 'восприяти', 570: 'ты ', 571: 'ети', 572: 'друго', 573: 'были ', 574: 'торо', 575: 'за ', 576: 'и в', 577: 'этого ', 578: 'тер', 579: 'я не ', 580: 'мира', 581: 'пр', 582: 'быть ', 583: 'лек', 584: 'согла', 585: 'ели', 586: 'по ', 587: 'кое ', 588: 'доро', 589: '. А', 590: 'кий ', 591: 'ёт ', 592: 'особ', 593: 'ению ', 594: 'гом ', 595: 'станов', 596: 'следу', 597: 'созда', 598: 'ка, ', 599: 'ори', 600: 'фо', 601: ': «', 602: 'холод', 603: 'люди ', 604: 'ительно ', 605: 'нт', 606: 'друг ', 607: 'этом ', 608: '.М', 609: 'стре', 610: 'образо', 611: 'кру', 612: 'друга', 613: 'гна', 614: 'критики', 615: 'критика ', 616: '\xa0— ', 617: 'нени', 618: 'времен', 619: 'аль', 620: 'А.С. Пушки', 621: 'два ', 622: 'а и у', 623: 'межд', 624: 'между ', 625: 'основ', 626: 'каза', 627: 'лё', 628: 'страда', 629: 'ния ', 630: 'торая ', 631: 'боль', 632: 'ях ', 633: 'вол', 634: 'всё ', 635: 'зрения ', 636: 'чувство', 637: ', по', 638: 'ар', 639: 'сть ', 640: '. Для ', 641: 'близ', 642: '.\nП', 643: 'поэ', 644: 'сущест', 645: 'свою ', 646: 'жизнь', 647: 'она ', 648: 'не по', 649: 'своей ', 650: 'л, ', 651: 'свои ', 652: 'чат', 653: '-то ', 654: 'ез', 655: 'читель', 656: 'ющ', 657: 'казывает ', 658: '. М', 659: 'ря', 660: 'которые ', 661: 'чер', 662: 'крас', 663: 'ходит', 664: 'зо', 665: 'счасть', 666: 'сае', 667: 'заклю', 668: 'пробле', 669: '.\nТа', 670: 'обра', 671: 'размы', 672: 'размыш', 673: '.\nИ', 674: 'становится ', 675: 'я по', 676: 'зици', 677: 'ельз', 678: 'ельзя не ', 679: 'исти', 680: 'экст', 681: 'экстре', 682: 'экстрема', 683: 'экстремаль', 684: 'ти ', 685: 'тя', 686: 'тяж', 687: 'ство ', 688: 'сли ', 689: ', а ', 690: 'чина', 691: 'глу', 692: 'сме', 693: 'ше', 694: ', когда ', 695: 'полож', 696: 'ют', 697: 'шь ', 698: 'или ', 699: 'пра', 700: 'боле', 701: 'тому ', 702: 'крыва', 703: 'и на', 704: 'смот', 705: 'м образо', 706: 'отношение ', 707: 'Достоевско', 708: '\xa0 ', 709: '\xa0 \n', 710: 'ут', 711: 'вать ', 712: 'еди', 713: 'человек ', 714: 'ен ', 715: '. Достоевс', 716: '.Н. ', 717: 'является ', 718: 'к животны', 719: 'гнато', 720: 'полы', 721: 'полынь', 722: 'уто', 723: 'ямщи', 724: 'роль ', 725: 'Напиш', 726: 'Напишите ', 727: 'Напишите со', 728: 'Напишите сочи', 729: 'Напишите сочинени', 730: 'Напишите сочинение', 731: 'Напишите сочинение-', 732: 'Напишите сочинение-расс', 733: 'Напишите сочинение-рассу', 734: 'Напишите сочинение-рассужд', 735: 'Напишите сочинение-рассуждение ', 736: 'Напишите сочинение-рассуждение на ', 737: 'Напишите сочинение-рассуждение на те', 738: 'Напишите сочинение-рассуждение на тему ', 739: '. Покажите ', 740: '. Покажите мнение ', 741: '. Покажите мнение автор', 742: '. Покажите мнение автора в ', 743: '. Покажите мнение автора в сво', 744: '. Покажите мнение автора в своем ', 745: '. Покажите мнение автора в своем текст', 746: '. Покажите мнение автора в своем тексте, ', 747: '. Покажите мнение автора в своем тексте, при', 748: '. Покажите мнение автора в своем тексте, привед', 749: '. Покажите мнение автора в своем тексте, приведите ', 750: '. Покажите мнение автора в своем тексте, приведите два ', 751: '. Покажите мнение автора в своем тексте, приведите два ф', 752: '. Покажите мнение автора в своем тексте, приведите два фраг', 753: '. Покажите мнение автора в своем тексте, приведите два фрагмент', 754: '. Покажите мнение автора в своем тексте, приведите два фрагмента и у', 755: '. Покажите мнение автора в своем тексте, приведите два фрагмента и укажите ', 756: '. Покажите мнение автора в своем тексте, приведите два фрагмента и укажите свя', 757: '. Покажите мнение автора в своем тексте, приведите два фрагмента и укажите связ', 758: '. Покажите мнение автора в своем тексте, приведите два фрагмента и укажите связь ', 759: '. Покажите мнение автора в своем тексте, приведите два фрагмента и укажите связь между ', 760: '. Покажите мнение автора в своем тексте, приведите два фрагмента и укажите связь между ни', 761: '. Покажите мнение автора в своем тексте, приведите два фрагмента и укажите связь между ними', 762: 'а основ', 763: 'а основе культур', 764: 'а основе культуры по', 765: 'а основе культуры пояс', 766: 'а основе культуры поясн', 767: 'а основе культуры поясните ', 768: 'а основе культуры поясните сво', 769: 'а основе культуры поясните свое ', 770: 'а основе культуры поясните свое мнение ', 771: 'а основе культуры поясните свое мнение к ', 772: 'а основе культуры поясните свое мнение к автор', 773: 'а основе культуры поясните свое мнение к авторско', 774: 'а основе культуры поясните свое мнение к авторскому', 775: 'а основе культуры поясните свое мнение к авторскому.|eof|\n', 776: 'любо', 777: 'ков ', 778: 'не зна', 779: 'ниц', 780: 'ный, ', 781: 'зем', 782: 'перь ', 783: '. Т', 784: 'нес', 785: 'ность', 786: 'недо', 787: 'ских ', 788: 'нят', 789: 'миро', 790: 'непосред', 791: 'непосредствен', 792: 'ха', 793: '; ', 794: 'нако ', 795: 'а не ', 796: 'общ', 797: 'себя ', 798: 'ет, ', 799: 'ум', 800: 'стро', 801: 'лся, ', 802: 'каждого ', 803: '. С', 804: 'вление ', 805: 'ётся ', 806: 'жел', 807: '? ', 808: 'менно ', 809: '– ', 810: 'ь, что ', 811: ', которы', 812: 'связа', 813: 'яет ', 814: 'ского ', 815: 'доби', 816: 'вить ', 817: 'тогом ', 818: 'тогом размыш', 819: 'тогом размышлени', 820: 'тогом размышлений ', 821: 'я позици', 822: 'я позиция', 823: 'я позиция: ', 824: 'ни в ', 825: 'ельзя не согла', 826: 'ельзя не согласиться ', 827: 'ельзя не согласиться с ', 828: 'ельзя не согласиться с точ', 829: 'ельзя не согласиться с точкой ', 830: 'ельзя не согласиться с точкой зрения ', 831: 'автора', 832: 'ятся ', 833: 'пример, ', 834: 'ет, что ', 835: 'чение ', 836: 'человека, ', 837: 'Тер', 838: 'Террит', 839: 'Территори', 840: 'ника', 841: 'том ', 842: 'полу', 843: 'была ', 844: 'верну', 845: 'нуж', 846: 'нут', 847: 'тё', 848: 'глав', 849: '. Но ', 850: 'е: ', 851: 'вопро', 852: 'ены ', 853: 'гие ', 854: 'писатель ', 855: 'сторо', 856: 'эти', 857: 'з ', 858: 'стрем', 859: 'отношение к ', 860: 'дву', 861: 'двух ', 862: 'жни', 863: '... ', 864: 'верш', 865: 'бро', 866: 'ения', 867: 'ые ', 868: 'можно ', 869: 'ересае', 870: 'писатель', 871: '.Н. Толстого ', 872: 'Ф.М', 873: 'Ф.М. Достоевс', 874: 'ит ', 875: 'ую ', 876: 'лась ', 877: 'ятель', 878: '. Од', 879: 'тиц', 880: 'не мо', 881: 'чь ', 882: 'гнатович ', 883: 'проце', 884: 'процесс', 885: 'не\xa0', 886: 'ность и ', 887: 'ак', 888: 'Бо', 889: 'всё', 890: '. Ка', 891: 'настоя', 892: 'настоящи', 893: 'ду ', 894: 'него ', 895: '.\nО', 896: 'теперь ', 897: 'ющий ', 898: ', которая ', 899: 'нами ', 900: 'больш', 901: 'он', 902: 'ческих ', 903: 'нял ', 904: 'и и ', 905: 'ны, ', 906: 'иде', 907: 'ры', 908: 'ень', 909: 'му, ', 910: 'риц', 911: 'ня ', 912: 'й, и ', 913: 'которо', 914: 'води', 915: 'оти', 916: 'дать ', 917: '.\nК', 918: 'того, ', 919: 'м о', 920: 'не только ', 921: 'кни', 922: 'стихи ', 923: 'как', 924: 'как-то ', 925: 'след', 926: 'лись в ', 927: 'коне', 928: ', он ', 929: 'поми', 930: 'помина', 931: 'чёр', 932: 'кива', 933: 'ке, ', 934: 'же, ', 935: 'зя', 936: 'елико', 937: 'воль', 938: 'самое ', 939: 'зам', 940: 'нам ', 941: 'без ', 942: 'куд', 943: 'втор', 944: 'бла', 945: 'благо', 946: 'ем, ', 947: 'ми, ', 948: 'исто', 949: 'ить и ', 950: 'красот', 951: 'чь', 952: 'люб', 953: 'всегда ', 954: 'вли', 955: '(П', 956: 'йх', 957: 'йхен', 958: 'йхенва', 959: 'йхенваль', 960: '*)', 961: '(1', 962: '(18', 963: '-1', 964: '-19', 965: ' — ', 966: 'заключа', 967: 'заключается ', 968: 'особен', 969: 'над ', 970: 'предлож', 971: 'предложен', 972: 'А.С. Пушкина ', 973: 'кры', 974: 'сколько ', 975: 'когда ', 976: 'автора ', 977: 'все ', 978: 'щается ', 979: 'связаны ', 980: 'заи', 981: 'м на', 982: 'произведения ', 983: '.\nИтогом размышлений ', 984: 'становится следу', 985: 'становится следую', 986: 'становится следующа', 987: 'становится следующая позиция: ', 988: 'оно ', 989: '.\nН', 990: '.\nНельзя не согласиться с точкой зрения ', 991: 'автора. Д', 992: 'автора. Дей', 993: 'автора. Действ', 994: 'автора. Действитель', 995: 'автора. Действительно, ', 996: 'например, ', 997: 'рома', 998: 'отли', 999: 'ными ', 1000: 'равно', 1001: 'равнодуш', 1002: 'дар', 1003: '. Покажите мнение автора в своем тексте, приведите два фрагмента и укажите связь между ними. Н', 1004: '. Покажите мнение автора в своем тексте, приведите два фрагмента и укажите связь между ними. На основе культуры поясните свое мнение к авторскому.|eof|\n', 1005: 'уз', 1006: 'очен', 1007: 'очень ', 1008: 'условия', 1009: '. Для этого ', 1010: '. Для этого по', 1011: '. Для этого пона', 1012: '. Для этого понадоби', 1013: 'кто ', 1014: '. Ч', 1015: 'Территории, ', 1016: 'зат', 1017: 'матер', 1018: 'матери', 1019: '. Не ', 1020: 'имя ', 1021: 'время ', 1022: 'рад', 1023: 'ради '}
