import speechmetrics as sm

if __name__ == '__main__':
    window = 5

    print('Trying ABSOLUTE metrics: ')
    metrics = sm.load('absolute', window)

    reference = 'data/m2_script1_produced.wav'
    tests = ['data/m2_script1_clean.wav',
             'data/m2_script1_ipad_confroom1.wav',
             'data/m2_script1_produced.wav']

    for test in tests:
        import pprint
        print('Computing scores for ', test)
        scores = metrics(reference, test)
        pprint.pprint(scores)

    print('\nTrying RELATIVE metrics: ')

    metrics = sm.load('relative', window)

    reference = 'data/m2_script1_produced.wav'
    tests = ['data/m2_script1_clean.wav',
             'data/m2_script1_ipad_confroom1.wav',
             'data/m2_script1_produced.wav']

    for test in tests:
        import pprint
        print('Computing scores for ', test)
        scores = metrics(reference, test)
        pprint.pprint(scores)
