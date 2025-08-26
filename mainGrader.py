import CSVgrader
import CSVsortbyscore
import GradingTweaking

if __name__ == '__main__':
    current_iteration = 6
    database = "PM"
    delimiter = ","

    input_path = 'Articles/'+database+'/Scored_articles/scored_graded_articles'+str(current_iteration)+'.csv'
    output_path = 'Articles/'+database+'/Graded_articles/graded_articles'+str(current_iteration+1)+'.csv'
    output_path_sorted = 'Articles/'+database+'/Graded_sorted_articles/graded_sorted_articles' + str(current_iteration + 1) + '.csv'

    print('Reading input file: ', input_path)
    print('Setting output file: ', output_path)

    # tweak config
    """
    GradingTweaking.pipeline_tuning(
        input_file_path=input_path,
        output_file_path='graded_articles_updated.csv',
        max_outer_iterations=20,
        monte_carlo_iterations=50,
        base_tweak_factor=0.1,
        patience=3,
        delimiter=delimiter
    )
    """

    # Grade CSV
    CSVgrader.score_articles(input_path, output_path, delimiter)

    # Sort CSV
    CSVsortbyscore.sort_csv_by_score(output_path, output_path_sorted)