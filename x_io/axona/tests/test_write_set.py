from x_io.axona.write_set (
    get_session_parameters
)

def test_get_session_parameters():
    session_parameters = get_session_parameters(n_channels=4)
    assert 
    counter = 0
    for value in session_parameters.values():
        if value == None:
            counter += 1
    print('{} keys not set'.format(counter))
