Feature: Running image statistics matching with different parameters

  Background: Mandatory parameters
    Given source image path
    And reference image path
    And resulting image path

  Scenario Outline: Calling with default parameters
    Given <matching_algorithm>
    When calling main
    Then no exception is raised
    And an output file is created

    Examples:
      | matching_algorithm |
      | hm                 |
      | fdm                |

  Scenario Outline: Calling with color space parameter
    Given <matching_algorithm>
    And <color_space_param>
    When calling main
    Then no exception is raised
    And an output file is created

    Examples:
      | color_space_param  | matching_algorithm |
      | --color-space hsv  | hm                 |
      | --color-space lab  | hm                 |
      | --color-space rgb  | hm                 |
      | -s hsv             | hm                 |
      | -s lab             | hm                 |
      | -s rgb             | hm                 |

  Scenario Outline: Calling with gray color space parameter
    Given <matching_algorithm>
    And gray source image path
    And gray reference image path
    And <color_space_param>
    And <channels_param>
    When calling main
    Then no exception is raised
    And an output file is created

    Examples:
      | color_space_param  | channels_param   | matching_algorithm |
      | --color-space gray | --channels 0     | hm                 |
      | -s gray            | -c 0             | hm                 |

  Scenario Outline: Calling with wrong color space parameter
    Given <matching_algorithm>
    And <color_space_param>
    When calling main
    Then an exception is raised

    Examples:
      | color_space_param | matching_algorithm |
      | --color-space YIQ | hm                 |
      | -s YIQ            | hm                 |


  Scenario Outline: Calling with channel parameter
    Given <matching_algorithm>
    And <channels_param>
    When calling main
    Then no exception is raised
    And an output file is created

    Examples:
      | channels_param   | matching_algorithm |
      | --channels 0     | hm                 |
      | --channels 1     | hm                 |
      | --channels 2     | hm                 |
      | --channels 0,1   | hm                 |
      | --channels 0,2   | hm                 |
      | --channels 1,2   | hm                 |
      | --channels 0,1,2 | hm                 |
      | -c 0             | hm                 |
      | -c 1             | hm                 |
      | -c 2             | hm                 |
      | -c 0,1           | hm                 |
      | -c 0,2           | hm                 |
      | -c 1,2           | hm                 |
      | -c 0,1,2         | hm                 |

  Scenario Outline: Calling with wrong channel parameter
    Given <matching_algorithm>
    And <channels_param>
    When calling main
    Then an exception is raised

    Examples:
      | channels_param     | matching_algorithm |
      | --channels -1      | hm                 |
      | --channels 3       | hm                 |
      | --channels 0,0     | hm                 |
      | --channels 0,1,2,3 | hm                 |
      | -c -1              | hm                 |
      | -c 3               | hm                 |
      | -c 0,0             | hm                 |
      | -c 0,1,2,3         | hm                 |

  Scenario Outline: Calling with match proportion parameter
    Given <matching_algorithm>
    And <match_proportion_param>
    When calling main
    Then no exception is raised
    And an output file is created

    Examples:
      | match_proportion_param   | matching_algorithm |
      | --match-proportion 0.    | hm                 |
      | --match-proportion 1.    | hm                 |
      | --match-proportion 0.567 | hm                 |
      | -m 0.                    | hm                 |
      | -m 1.                    | hm                 |
      | -m 0.567                 | hm                 |

  Scenario Outline: Calling with verify input parameter
    Given <matching_algorithm>
    And <verify_input_param>
    When calling main
    Then no exception is raised
    And an output file is created

    Examples:
      | verify_input_param      | matching_algorithm |
      | --verify-input False    | hm                 |
      | --verify-input True     | hm                 |
      | -v False                | hm                 |
      | -v True                 | hm                 |

  Scenario Outline: Calling with plot parameter
    Given <matching_algorithm>
    And <plot_param>
    When calling main
    Then no exception is raised
    And an HM plot file is created

    Examples:
      | plot_param | matching_algorithm |
      | --plot     | hm                 |
      | -p         | hm                 |

  Scenario Outline: Calling fdm with match proportion parameter
    Given <matching_algorithm>
    And <match_proportion_param>
    When calling main
    Then an exception is raised

    Examples:
      | match_proportion_param | matching_algorithm |
      | --match-proportion 0.5 | fdm                |

  Scenario Outline: Calling with wrong match proportion parameter
    Given <matching_algorithm>
    And <match_proportion_param>
    When calling main
    Then an exception is raised

    Examples:
      | match_proportion_param  | matching_algorithm |
      | --match-proportion -0.1 | hm                 |
      | --match-proportion 1.1  | hm                 |

  Scenario Outline: Calling with non-existing parameter
    Given <matching_algorithm>
    And <channels_param>
    When calling main
    Then an exception is raised

    Examples:
      | channels_param      | matching_algorithm |
      | --non_existing 1337 | hm                 |
