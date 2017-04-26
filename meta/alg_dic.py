'''
   fmm-gen    
   Generating Families of Practical Fast Matrix Multiplication Algorithms

   Copyright (C) 2017, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

alg_dic = {
'fast323' : [ '../meta/algorithms/grey323-15-103', ( 3, 2, 3 ) ],
'fast424' : [ '../meta/algorithms/grey424-26-257', ( 4, 2, 4 ) ],
# Mapping table for all possible partitions: 40+ FMM algorithms
'classical222_8_24'             : [  '../meta/algorithms/classical222-8-24'           , ( 2, 2, 2 ) ],
'classical234_24_72'            : [  '../meta/algorithms/classical234-24-72'          , ( 2, 3, 4 ) ],
'classical333_27_81'            : [  '../meta/algorithms/classical333-27-81'          , ( 3, 3, 3 ) ],
'classical423_24_72'            : [  '../meta/algorithms/classical423-24-72'          , ( 4, 2, 3 ) ],
'fast423_130'                   : [  '../meta/algorithms/fast423-130'                 , ( 4, 2, 3 ) ],
'fast423_134'                   : [  '../meta/algorithms/fast423-134'                 , ( 4, 2, 3 ) ],
'fast423_138'                   : [  '../meta/algorithms/fast423-138'                 , ( 4, 2, 3 ) ],
'fast423_156'                   : [  '../meta/algorithms/fast423-156'                 , ( 4, 2, 3 ) ],
'grey234_20_144'                : [  '../meta/algorithms/grey234-20-144'              , ( 2, 3, 4 ) ],
'grey243_20_144'                : [  '../meta/algorithms/grey243-20-144'              , ( 2, 4, 3 ) ],
'grey252_18_99'                 : [  '../meta/algorithms/grey252-18-99'               , ( 2, 5, 2 ) ],
'grey322_11_50'                 : [  '../meta/algorithms/grey322-11-50'               , ( 3, 2, 2 ) ],
'grey323_15_103'                : [  '../meta/algorithms/grey323-15-103'              , ( 3, 2, 3 ) ],
'grey324_20_144'                : [  '../meta/algorithms/grey324-20-144'              , ( 3, 2, 4 ) ],
'grey332_15_103'                : [  '../meta/algorithms/grey332-15-103'              , ( 3, 3, 2 ) ],
'grey333_23_142'                : [  '../meta/algorithms/grey333-23-142'              , ( 3, 3, 3 ) ],
'grey333_23_143'                : [  '../meta/algorithms/grey333-23-143'              , ( 3, 3, 3 ) ],
'grey333_23_144'                : [  '../meta/algorithms/grey333-23-144'              , ( 3, 3, 3 ) ],
'grey333_23_152'                : [  '../meta/algorithms/grey333-23-152'              , ( 3, 3, 3 ) ],
'grey333_23_221'                : [  '../meta/algorithms/grey333-23-221'              , ( 3, 3, 3 ) ],
'grey342_20_144'                : [  '../meta/algorithms/grey342-20-144'              , ( 3, 4, 2 ) ],
'grey343_29_234'                : [  '../meta/algorithms/grey343-29-234'              , ( 3, 4, 3 ) ],
'grey422_14_84'                 : [  '../meta/algorithms/grey422-14-84'               , ( 4, 2, 2 ) ],
'grey423_20_144'                : [  '../meta/algorithms/grey423-20-144'              , ( 4, 2, 3 ) ],
'grey424_26_257'                : [  '../meta/algorithms/grey424-26-257'              , ( 4, 2, 4 ) ],
'grey432_20_144'                : [  '../meta/algorithms/grey432-20-144'              , ( 4, 3, 2 ) ],
'grey433_29_234'                : [  '../meta/algorithms/grey433-29-234'              , ( 4, 3, 3 ) ],
'grey442_26_257'                : [  '../meta/algorithms/grey442-26-257'              , ( 4, 4, 2 ) ],
'grey522_18_99'                 : [  '../meta/algorithms/grey522-18-99'               , ( 5, 2, 2 ) ],
'grey_strassen'                 : [  '../meta/algorithms/grey-strassen'               , ( 2, 2, 2 ) ],
'hk323_15_94'                   : [  '../meta/algorithms/hk323-15-94'                 , ( 3, 2, 3 ) ],
'hk332_15_94'                   : [  '../meta/algorithms/hk332-15-94'                 , ( 3, 3, 2 ) ],
'madan232_11_48'                : [  '../meta/algorithms/madan232-11-48'              , ( 2, 3, 2 ) ],
'smirnov333_23_139'             : [  '../meta/algorithms/smirnov333-23-139'           , ( 3, 3, 3 ) ],
'smirnov336_40_960'             : [  '../meta/algorithms/smirnov336-40-960'           , ( 3, 3, 6 ) ],
'smirnov343_29_204'             : [  '../meta/algorithms/smirnov343-29-204'           , ( 3, 4, 3 ) ],
'smirnov353_36_280'             : [  '../meta/algorithms/smirnov353-36-280'           , ( 3, 5, 3 ) ],
'smirnov363_40_960'             : [  '../meta/algorithms/smirnov363-40-960'           , ( 3, 6, 3 ) ],
'smirnov633_40_960'             : [  '../meta/algorithms/smirnov633-40-960'           , ( 6, 3, 3 ) ],
'strassen'                      : [  '../meta/algorithms/strassen'                    , ( 2, 2, 2 ) ],
# Approximate Algorithms (APA) or duplicated
'bini322_10_52_approx'          : [  '../meta/algorithms/ignore/bini322-10-52-approx'        , ( 3, 2, 2 ) ],
'schonhage333_21_117_approx'    : [  '../meta/algorithms/ignore/schonhage333-21-117-approx'  , ( 3, 3, 3 ) ],
'smirnov323_14_108_approx'      : [  '../meta/algorithms/ignore/smirnov323-14-108-approx'    , ( 3, 2, 3 ) ],
'smirnov333_20_182_approx'      : [  '../meta/algorithms/ignore/smirnov333-20-182-approx'    , ( 3, 3, 3 ) ],
'grey323_15_89'                 : [  '../meta/algorithms/ignore/grey323-15-89'               , ( 3, 2, 3 ) ],
# Mapping table for pure partitions: 20+ FMM algorithms
'222'  : [  '../meta/algorithms/strassen'                                 ,  ( 2, 2, 2 ) ],
'232'  : [  '../meta/algorithms/madan232-11-48'                           ,  ( 2, 3, 2 ) ],
'234'  : [  '../meta/algorithms/grey234-20-144'                           ,  ( 2, 3, 4 ) ],
'243'  : [  '../meta/algorithms/grey243-20-144'                           ,  ( 2, 4, 3 ) ],
'252'  : [  '../meta/algorithms/grey252-18-99'                            ,  ( 2, 5, 2 ) ],
'322'  : [  '../meta/algorithms/grey322-11-50'                            ,  ( 3, 2, 2 ) ],
'323'  : [  '../meta/algorithms/hk323-15-94'                              ,  ( 3, 2, 3 ) ],
'324'  : [  '../meta/algorithms/grey324-20-144'                           ,  ( 3, 2, 4 ) ],
'332'  : [  '../meta/algorithms/hk332-15-94'                              ,  ( 3, 3, 2 ) ],
'333'  : [  '../meta/algorithms/smirnov333-23-139'                        ,  ( 3, 3, 3 ) ],
'336'  : [  '../meta/algorithms/smirnov336-40-960'                        ,  ( 3, 3, 6 ) ],
'342'  : [  '../meta/algorithms/grey342-20-144'                           ,  ( 3, 4, 2 ) ],
'343'  : [  '../meta/algorithms/smirnov343-29-204'                        ,  ( 3, 4, 3 ) ],
'353'  : [  '../meta/algorithms/smirnov353-36-280'                        ,  ( 3, 5, 3 ) ],
'363'  : [  '../meta/algorithms/smirnov363-40-960'                        ,  ( 3, 6, 3 ) ],
'422'  : [  '../meta/algorithms/grey422-14-84'                            ,  ( 4, 2, 2 ) ],
'423'  : [  '../meta/algorithms/fast423-138'                              ,  ( 4, 2, 3 ) ],
'424'  : [  '../meta/algorithms/grey424-26-257'                           ,  ( 4, 2, 4 ) ],
'432'  : [  '../meta/algorithms/grey432-20-144'                           ,  ( 4, 3, 2 ) ],
'433'  : [  '../meta/algorithms/grey433-29-234'                           ,  ( 4, 3, 3 ) ],
'442'  : [  '../meta/algorithms/grey442-26-257'                           ,  ( 4, 4, 2 ) ],
'522'  : [  '../meta/algorithms/grey522-18-99'                            ,  ( 5, 2, 2 ) ],
'633'  : [  '../meta/algorithms/smirnov633-40-960'                        ,  ( 6, 3, 3 ) ],
}

#alg_set=[
#'222',
#'232',
#'234',
#'243',
#'252',
#'322',
#'323',
#'324',
#'332',
#'333',
#'336',
#'342',
#'343',
#'353',
#'363',
#'422',
#'423',
#'424',
#'432',
#'433',
#'442',
#'522',
#'633',
#]
#
#alg_set=[
#'strassen',
#'madan232_11_48',
#'grey234_20_144',
#'grey243_20_144',
#'grey252_18_99',
#'grey322_11_50',
#'hk323_15_94',
#'grey324_20_144',
#'hk332_15_94',
#'smirnov333_23_139',
#'smirnov336_40_960',
#'grey342_20_144',
#'smirnov343_29_204',
#'smirnov353_36_280',
#'smirnov363_40_960',
#'grey422_14_84',
#'fast423_138',
#'grey424_26_257',
#'grey432_20_144',
#'grey433_29_234',
#'grey442_26_257',
#'grey522_18_99',
#'smirnov633_40_960',
#]
