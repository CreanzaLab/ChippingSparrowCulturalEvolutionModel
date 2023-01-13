import pandas as pd


def find_best_models(path, learning_type):
    # number of recordings
    numRec_fisher_df = pd.read_csv(path + 'Hist_NumRecVsPercentTypes/' +
                                   'Collapsed_optimizedBins/' +
                                   learning_type +
                                   '_degError_numRec_fisher_df.csv')
    numRec_fisher_df.rename(columns={'pval': 'numRecPval'}, inplace=True)
    numRec_fisher_df['numRecPvalRank'] = numRec_fisher_df[
        'numRecPval'].rank(ascending=False)

    # lifespans
    lifespan_fisher_df = pd.read_csv(path + 'Hist_LifespansVsPercentTypes/' +
                                     'Collapsed_optimizedBins/' +
                                     learning_type +
                                     '_degError_lifespan_fisher_df.csv')
    lifespan_fisher_df.rename(columns={'pval': 'lifespanPval'}, inplace=True)
    lifespan_fisher_df['lifespanPvalRank'] = lifespan_fisher_df[
        'lifespanPval'].rank(ascending=False)

    fisher_df = pd.merge(numRec_fisher_df, lifespan_fisher_df, on=['disp',
                                                                   'error'])

    fisher_df['combinedRank'] = fisher_df['numRecPvalRank'] + fisher_df[
        'lifespanPvalRank']

    model_w_maxP = fisher_df.loc[fisher_df['combinedRank'].idxmin()]
    model_disp = model_w_maxP.disp
    model_error = model_w_maxP.error

    return [str(model_disp), str(model_error)]

