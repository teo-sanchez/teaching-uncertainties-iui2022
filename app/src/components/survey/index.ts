import { Survey, SurveyOptions, SurveyData, SurveyEntry, SurveyAnswer, SurveyQuestion } from './survey.component';

export function survey(data: SurveyData, options?: Partial<SurveyOptions>): Survey {
  return new Survey(data, options);
}

export type { Survey, SurveyOptions, SurveyData, SurveyEntry, SurveyAnswer, SurveyQuestion };
