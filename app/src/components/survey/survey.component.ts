import { 
  Component, 
  Stream, 
} from '@marcellejs/core';
import View from './survey.view.svelte';
import { VerticalSliderOptions } from '../../types';


/** * explanation: 
 *  Text displayed below the text of the question. Should explain what to do, e.g.: 'Choose between the following assessments.'
 *  If a function, will be passed the options so that a custom message can be displayed, e.g: 'Maximum ${options.open.max_char} characters.'
 */
export interface SurveyQuestionOptions {
  explanation: string | ((a: SurveyQuestionOptions) => string);
  skippable: boolean;
  skip: {
    text: string;
  };
  // Per-format options
  likert: {
    levels: number | string[];
    default_level: number;
  };
  text: {
    max_length: number;   
    placeholder: string;
  };
  open: {
  };
  radio: {
    values: string[];
    name: string;
    type: string;
  };
}

export interface SurveyQuestion {
  id: string | number;
  text: string;
  format: 'likert' | 'text' | 'open' | 'radio';
  options?: Partial<SurveyQuestionOptions>;
}

export interface SurveyAnswer {
  id: string | number;
  format: 'likert' | 'text' | 'open' | 'radio';
  skipped: boolean;
  data: {
    // TODO (before going for the real study)
    // put each in the sub-object of the corresponding format
    // in order to match the question options.
    level?: number; // likert
    text: string; // text
    value?: string; // radio
  };
}

export interface SurveyEntry {
  id: string | number;
  q: SurveyQuestion;
  a: SurveyAnswer;
}

export interface SurveyData {
  default: {
    options: Partial<SurveyQuestionOptions>;
  };
  questions: SurveyQuestion[];
}

export interface SurveyOptions {
  slider: Partial<VerticalSliderOptions>;
}

export class Survey extends Component {
  title = 'survey [custom module 🤖]';

  $options: Stream<SurveyOptions>;

  $questions: Stream<Array<SurveyQuestion>>;
  $answers: Stream<Array<SurveyAnswer>>;
  $entries: Stream<Array<SurveyEntry>>;

  constructor(data: SurveyData, options?: Partial<SurveyOptions>) {
    super();
    this.$options = new Stream({
      slider: {
        // see options for slider Svelte component etc.
        // here: https://simeydotme.github.io/svelte-range-slider-pips/
        springValues: {
          stiffness: 0.15,
          damping: 0.8,
          ...options?.slider?.springValues
        },
      },
      ...options
    }, true);
    this.$options.start();

    const questions = data.questions.map(question => ({
      ...question,
      options: {
        ...question.options,
        ...data.default.options
      }
    } as SurveyQuestion));
    const answers = questions.map(question => {
      const levels = question.options?.likert?.levels;
      const numberLevels = Array.isArray(levels) ? (levels as string[]).length : levels;
      const defaultLevel = question.options?.likert?.default_level
            ? question.options?.likert?.default_level
            : Math.floor(numberLevels / 2);
      return {
        id: question.id,
        format: question.format,
        skipped: false,
        data: {
         level: defaultLevel,
         text: null 
        },
      } as SurveyAnswer;
    });
    this.$questions = new Stream(questions, true);
    this.$questions.start();
    this.$answers = new Stream(answers, true);
    this.$answers.start();

    this.$entries = this.$answers.combine(
      (questions: SurveyQuestion[], answers: SurveyAnswer[]) => 
        questions.map((question, idx) => 
        ({
          id: question.id,
          q: question,
          a: answers[idx]
        } as SurveyEntry)),
        this.$questions
    );
    this.$entries.start();
  }

  mount(target?: HTMLElement): void {
    const t = target || document.querySelector(`#${this.id}`);
    if (!t) return;
    this.destroy();
    this.$$.app = new View({
      target: t,
      props: {
        title: this.title,
        questions: this.$questions,
        answers: this.$answers,
        options: this.$options,
      },
    });
  }

}
