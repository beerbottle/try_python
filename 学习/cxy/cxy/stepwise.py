  
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging
import os
from scipy.stats import chi2
  def stepwise(self, X, y, logging_file, start_from=[], direction='FORWARD',
                    lrt=True, lrt_threshold=0.05):
        """
        Use results of logistic regression to select features. It will go through
        stepwise based on the direction specified first and then check the
        significance of the variables using pvalue. Then check if there are variables
        with negative coefficients and remove it.

        Args:
        X (pd.DataFrame()): X data frame that contains the variables.
            variable names should not contain . or -, replace with _ and _
        y (pd.Series()): y data, labeling the performance
        logging_file (str): logging file object
        start_from (list): list of variable names that the model starts from,
            if not the null model
        direction (str): default='FOWARD/BACKWARD', forward first and then backward.
            ['FORWARD/BACKWARD', 'BACKWARD/FORWARD']
        lrt (bool): default = True. If set true, the model will consider likelihood
            ratio test result
        lrt_threshold (float): default = 0.05. The alpha value for likelihood ratio
            test.

        Returns:
        imp_df (pd.DataFrame()): contains the variable names and FORWARD AIC
            and ranking
        """
        def _likelihood_ratio_test(ll_r, ll_f, lrt_threshold):
            # H0: reduced model is true. If rejected, then it means that the
            # full model is good. Need to add the candidate
            test_statistics = (-2 * ll_r) - (-2 * ll_f)
            p_value = 1 - chi2(df=1).cdf(test_statistics)
            return p_value <= lrt_threshold




        def _forward(dataset, logging_file, current_score, best_new_score,
                    remaining, selected, result_dict, reduced_loglikelihood):
            # TODO: 加上 f test like in SAS
            logging_file.info("While loop beginning current_score: %s" % current_score)
            logging_file.info("While loop beginning best_new_score: %s" % best_new_score)
            current_score = best_new_score
            aics_with_candidates = {}
            p_values_ok_to_add = []
            # 选择最好的变量to add
            for candidate in remaining:
                formula = "{} ~ {}".format('y', ' + '.join(selected + [candidate]))
                mod1 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
                # 只有新加指标的coefficient>0或者加上之后其他指标coefficient也>0
                if sum(mod1.params.loc[mod1.params.index != 'Intercept'] < 0) == 0 :
                    aics_with_candidates[candidate] = mod1.aic
                    full_loglikelihood = mod1.llf
                    if lrt:
                        p_values_ok = _likelihood_ratio_test(reduced_loglikelihood, full_loglikelihood, lrt_threshold)
                        if p_values_ok:
                            p_values_ok_to_add.append(candidate)

            # 只有通过likelihood ratio test的变量才会进行AIC比较进行选择
            candidate_scores = pd.Series(aics_with_candidates)
            if lrt:
                candidate_scores = candidate_scores.loc[p_values_ok_to_add]

            # 有变量pvalues显著 reject the reduced model and need to add the variable
            if not candidate_scores.empty:
                best = candidate_scores[candidate_scores == candidate_scores.min()]
                # best_new_score 被替换成新的加上这个变量的模型的AIC
                best_new_score = best.iloc[0]
                best_candidate = best.index.values[0]
            else:
                return None

            # 当加上变量的模型的AIC比当前模型的小时，选择加上变量的模型
            if current_score > best_new_score:
                logging_file.info('Best Variable to Add: %s' % best_candidate)
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                improvement_gained = current_score - best_new_score
                result_dict[best_candidate] = {'AIC_delta': improvement_gained, 'step': 'FORWARD'}
                formula = "{} ~ {}".format('y', ' + '.join(selected))
                mod2 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
                # loglikelihood of the reduced model
                reduced_loglikelihood = mod2.llf
                logging_file.info('FOWARD Step: AIC=%s' % mod2.aic)
                logging_file.info(mod2.summary())
                return current_score, best_new_score, result_dict, selected, remaining, reduced_loglikelihood
            else:
                return None



        def _backward(dataset, logging_file, current_score, best_new_score,
                    remaining, selected, result_dict, reduced_loglikelihood):
            # TODO: 加上 f test like in SAS
            logging_file.info("While loop beginning current_score: %s" % current_score)
            logging_file.info("While loop beginning best_new_score: %s" % best_new_score)
            current_score = best_new_score
            aics_with_candidates = {}
            p_values_ok_to_delete = []
            # 选择最差的to delete
            for candidate in selected:
                put_in_model = [i for i in selected if i != candidate]
                formula = "{} ~ {}".format('y', ' + '.join(put_in_model))
                mod1 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
                # 只有减去指标后留下的变量中没有变量coefficient是负数
                if sum(mod1.params.loc[mod1.params.index != 'Intercept'] < 0) == 0 :
                    aics_with_candidates[candidate] = mod1.aic
                    reduced_reduced_loglikelihood = mod1.llf
                    if lrt:
                        p_values_rejected = _likelihood_ratio_test(reduced_reduced_loglikelihood, reduced_loglikelihood, lrt_threshold)
                        # if not rejected, H0: reduced model is true, need to remove the variable
                        if not p_values_rejected:
                            p_values_ok_to_delete.append(candidate)

            # 只有通过likelihood ratio test的变量才会进行AIC比较进行选择
            candidate_scores = pd.Series(aics_with_candidates)
            if lrt:
                candidate_scores = candidate_scores.loc[p_values_ok_to_delete]

            # 有变量pvalues不显著 没有reject the reduced model, then need to delete the variable
            if not candidate_scores.empty:
                best = candidate_scores[candidate_scores == candidate_scores.min()]
                # best_new_score 被替换成新的减去这个变量的模型的AIC
                best_new_score = best.iloc[0]
                best_candidate = best.index.values[0]
            else:
                return None


            # 当减去变量的模型的AIC比当前模型的小时，选择减去变量的模型
            if current_score > best_new_score:
                logging_file.info('Best Variable to Delete: %s' % best_candidate)
                remaining.append(best_candidate)
                selected.remove(best_candidate)
                improvement_gained = current_score - best_new_score
                result_dict[best_candidate] = {'AIC_delta': improvement_gained, 'step': 'BACKWARD'}
                formula = "{} ~ {}".format('y', ' + '.join(selected))
                mod2 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
                # loglikelihood of the reduced model
                reduced_loglikelihood = mod2.llf
                logging_file.info('BACKWARD Step: AIC=%s' % mod2.aic)
                logging_file.info(mod2.summary())
                return current_score, best_new_score, result_dict, selected, remaining, reduced_loglikelihood
            else:
                return None