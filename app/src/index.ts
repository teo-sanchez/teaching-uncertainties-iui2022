import {
  webcam,
  datasetBrowser,
  mobileNet,
  dataset,
  button,
  dataStore,
  dashboard,
  trainingProgress,
  imageDisplay,
  Stream,
  Instance,
  Dataset,
  Prediction,
  notification,
  select,
  textField,
  text,
  DatasetBrowser,
  Dashboard,
  Button,
  logger,
} from '@marcellejs/core';

import '@marcellejs/core/marcelle.css';

import {
  deepEnsembleMlp,
  sliderChart,
  classificationLabel,
  survey,
  SurveyData,
  SurveyEntry,
  timer,
  buttonSelect,
  sliderArray,
  SliderChartData,
  pythonClassifier,
  PythonClassifierOptions,
  Survey,
} from './components';

import { MyService, myService, getIdService, Message } from './services';

import { UncertaintyUtility, ImageUtility, Utility } from './util';
import { EnsembleResults, DensityEstimatorResults, Uncertainty, VerticalSliderOptions } from './types';

// -----------------------------------------------------------
// SOME CONFIG
// -----------------------------------------------------------

import data from './assets/config/data.json';
import script from './assets/config/text_en.json';

import backend from '../backend/config/default.json';
import pythonBackend from '../backend/config/python.json';

// IMAGES
const accuracyTestImageUrls: string[] = [];
const accuracyTestImageLabels: string[] = [];

import nineImageModules from "./assets/images/user_test_accuracy/nine/*.jpg";
import queenImageModules from "./assets/images/user_test_accuracy/queen/*.jpg";
import kingImageModules from "./assets/images/user_test_accuracy/king/*.jpg";

const imageModules: Record<string, any> = {
  'Neuf': nineImageModules,
  'Dame': queenImageModules,
  'Roi': kingImageModules,
};

Object.entries(imageModules).map(([label, mod]) => {
  const imageUrls = mod.map((el: any) => el.default) as string[]
  const imageLabels = new Array<string>(imageUrls.length).fill(label);
  accuracyTestImageUrls.push(...imageUrls);
  accuracyTestImageLabels.push(...imageLabels);
});
import uncertaintyTestImageModules from "./assets/images/user_test_uncertainty/*.jpg";
const uncertaintyTestImageUrls = uncertaintyTestImageModules.map((el: any) => el.default) as string[];

type UncertaintyLabel = 'aleatoric' | 'epistemic';
type Condition = 'a' | 'b';

const conditions = data.conditions as Condition[];
const conditionUncertainties = data.condition_uncertainties as Record<Condition, UncertaintyLabel>;
const iterations = new Array(conditions.length).fill(0).map((_, idx) => idx.toString());

type AccuracyTestResult = 'dont_know' | 'right' | 'over' | 'under';
type AccuracyTestEntry = {
  answer: {
    value: string,
  },
  labels: {
    truth: string,
    predicted: string
  }
  result: AccuracyTestResult;
  image: {
    name: string;
    idx: number;
    order: number;
  }
}

type UncertaintyTestEntry = {
  answer: Partial<Uncertainty>;
  truth: Uncertainty;
  uncertainty: UncertaintyLabel;
  image: {
    name: string;
    idx: number;
    order: number;
  }
}

type ModelResults = {
  ensemble: EnsembleResults,
  gmm: DensityEstimatorResults,
}

// fixme: most 'server' operations fail the first time they're called.
// so we just do it twice.
async function force<T>(f: () => Promise<T>): Promise<T> {
  let res;
  try {
    res = await f();
  }
  catch {
    res = await f();
  }
  return res;
}

function getIFrameHtml(
  url: string,
  title: string,
  width: number,
  height: number,
): string {
  return `<iframe src='${url}?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479' 
        width='${width}' height='${height}' 
        frameborder='0' allow='autoplay; fullscreen; picture-in-picture' allowfullscreen 
        title='${title}'></iframe>`
}

function getAccuracyScore(condition: Condition, iteration: number, accuracyEntries: AccuracyTestEntry[]) {
  const accuracyScore = accuracyEntries.reduce((acc, entry) => {
    let score = 0;
    switch (entry.result) {
      case 'right': score = 1; break;
      case 'over': case 'under': score = -1; break;
      case 'dont_know': score = 0; break;
    }
    return acc + score;
  }, 0) / accuracyEntries.length;
  const numberOverestimate = accuracyEntries.reduce((acc, entry) => acc + (entry.result === 'over' ? 1 : 0), 0);
  const numberUnderestimate = accuracyEntries.reduce((acc, entry) => acc + (entry.result === 'under' ? 1 : 0), 0);
  return {
    condition,
    iteration,
    mark: accuracyScore * 10 + 10,
    numberOverestimate, numberUnderestimate
  };
}

const sliderOptions = <Record<Condition, Partial<VerticalSliderOptions>>>Object.fromEntries(conditions.map(condition =>
  [
    condition,
    {
      ...data.sliders.common as Partial<VerticalSliderOptions>,
      colour: (data.sliders[conditionUncertainties[condition]] as any).colour,
      formatter: (value: number) => value > 0.5
        ? script.misc.uncertainty_values[conditionUncertainties[condition] as keyof typeof script.misc.uncertainty_values].high
        : script.misc.uncertainty_values[conditionUncertainties[condition] as keyof typeof script.misc.uncertainty_values].low
    }
  ]
));

// -----------------------------------------------------------
// INPUT PIPELINE & DATA CAPTURE
// -----------------------------------------------------------


const storeUrl = `http://${backend.host}:${backend.port}`;
const gmmUrl = `ws://${pythonBackend.host}:${pythonBackend.port}${pythonBackend.namespace}`;
// const storeUrl = 'localStorage';
const store = dataStore({
  location: storeUrl,
  socket: {
    reconnectionAttempts: data.store.reconnectionAttempts,
    timeout: data.store.timeout // ms
  }
});
const storeConnectionPromise = store.connect();

let set: Dataset<number[][], string>;
let brows: DatasetBrowser<number[][]>;
const deepEnsemble = deepEnsembleMlp(
  {
    dataStore: store,
    numberModels: data.models.deep_ensemble.number_models,
  },
  {
    dataStore: store,
    layers: data.models.deep_ensemble.mlp.layers,
    epochs: data.models.deep_ensemble.mlp.epochs,
    validationSplit: data.models.deep_ensemble.mlp.validation_split,
  },
);
const densityClassifier = pythonClassifier({
  dataStore: store,
  modelType: data.models.gmm.model_type as PythonClassifierOptions["modelType"],
  parameters: {
    // max_iter: data.models.gmm.number_iterations,
    // covariance_type: data.models.gmm.covariance_type,
    scaler_path: data.models.gmm.scaler_path_relative
  },
});
const gmmConnectionPromise = densityClassifier.connect(gmmUrl);

const featureExtractor = mobileNet();


// -----------------------------------------------------------
// SERVICES
// -----------------------------------------------------------

let accuracyTestServices: Record<Condition, MyService<Array<AccuracyTestEntry>>>;
let uncertaintyTestService: MyService<Array<UncertaintyTestEntry>>;
let surveyService: MyService<Array<SurveyEntry>>;

// -----------------------------------------------------------
// DASHBOARDS
// -----------------------------------------------------------

const dashboards = <Record<Condition, Dashboard>>Object.fromEntries(conditions
  .map(condition => {
    const dash = dashboard({
      title: script.misc.dashboards[condition as keyof typeof script.misc.dashboards].name,
      author: script.misc.author.name
    });
    return [condition, dash];
  })
);

// -----------------------------------------------------------
// HOME
// -----------------------------------------------------------

const homeInputId = textField();
homeInputId.title = script.home.id_title;
const homeButton = button({ text: script.home.button_text });
homeButton.title = script.misc.pages.home.name;

const conditionSelect = select({ options: conditions });
conditionSelect.title = script.home.condition_title;
const iterationSelect = select({ options: iterations });
iterationSelect.title = script.home.iteration_title;

const $participantId = homeInputId.$text;
const $condition = conditionSelect.$value as Stream<Condition>;
const $iteration = iterationSelect.$value.map(val => parseInt(val)) as Stream<number>;

const homeDashboard = dashboard({
  title: script.misc.dashboards.home.name,
  author: script.misc.author.name
});

homeDashboard
  .page(script.misc.pages.home.name, false)
  .use(homeInputId, conditionSelect, iterationSelect, homeButton);


// -----------------------------------------------------------
// INTRO + INSTRUCTIONS
// -----------------------------------------------------------

const videoPrimer = text({
  text: getIFrameHtml(
    data.videos.primer.url,
    data.videos.primer.title,
    data.videos.width,
    data.videos.height
  )
});
videoPrimer.title = script.instructions.primer_video_title;
const videoInterface = text({
  text: getIFrameHtml(
    data.videos.interface.url,
    data.videos.interface.title,
    data.videos.width,
    data.videos.height
  )
});
videoInterface.title = script.instructions.interface_video_title;

// -----------------------------------------------------------
// TRAINING
// -----------------------------------------------------------

const cam = webcam({
  width: data.image.display_width,
  height: data.image.display_height,
  period: data.webcam.period
});
cam.title = script.teaching.webcam_title;

const trainingLabelSelect = buttonSelect(
  new Stream(data.labels.map((label: string) => script.misc.label_names[label as keyof typeof script.misc.label_names]), true),
  new Stream(data.labels, true)
);
trainingLabelSelect.title = script.teaching.snapshot_title;


const $features = cam.$images
  .map(async (img) => featureExtractor.process(img))
  .awaitPromises<number[][]>();
let ensembleIsProcessing = false;
const $ensembleResults = $features
  .filter(_ => !ensembleIsProcessing)
  .map(async (features) => {
    ensembleIsProcessing = true;
    const pred = await deepEnsemble.predict(features)
    ensembleIsProcessing = false;
    return pred;
  })
  .awaitPromises<EnsembleResults>();
let gmmIsProcessing = false;
const $gmmResults = $features
  .filter(_ => !gmmIsProcessing)
  .map(async (features) => {
    gmmIsProcessing = true;
    const pred = await densityClassifier.predict(features);
    gmmIsProcessing = false;
    return pred;
  })
  .awaitPromises<DensityEstimatorResults>();

const $uncertainty = $ensembleResults.combine(
  (gmmResult, ensembleResult) => ({
    gmm: gmmResult,
    ensemble: ensembleResult,
  } as ModelResults),
  $gmmResults
)
  .filter(result => result.ensemble != null && result.gmm != null)
  .map(result => UncertaintyUtility.computeUncertainty(result.ensemble, result.gmm))


const $labelPred = $ensembleResults
  .filter(result => result != null)
  .map(results => ({
    instanceId: '',
    label: results.label,
    confidences: results.confidences
  } as Prediction)
  );

const trainingProg = trainingProgress(deepEnsemble);
trainingProg.title = script.teaching.training_progress_title;

const time = timer();
time.title = script.teaching.timer_title;

const timerButton = button({ text: script.teaching.timer_button_text });
timerButton.title = script.teaching.timer_button_title;

const labelPrediction = classificationLabel($labelPred);
labelPrediction.title = script.teaching.label_title;

const trainingUncertaintyCharts = Object.fromEntries(conditions
  .map(condition => {
    const uncertainty = conditionUncertainties[condition] as keyof Uncertainty;
    const chart = sliderChart(
      $uncertainty.map(value => [{
        x: script.misc.uncertainty_names[uncertainty as keyof typeof script.misc.uncertainty_names],
        y: value[uncertainty],
        colour: (data.sliders[uncertainty as keyof typeof data.sliders] as any).colour
      } as SliderChartData]),
      sliderOptions[condition],
    );
    chart.title = script.teaching.uncertainty_title;
    return [condition, chart];
  })
);

/*
// -----------------------------------------------------------
// TRIAL
// -----------------------------------------------------------

const trialStore = dataStore();

const trialMLP = mlpClassifier({
  layers: [32,32],
  epochs: 10,
  dataStore: trialStore
});

const trialLabelSelect = buttonSelect({
  items: script.misc.labels
});

trialLabelSelect.title = script.teaching.snapshot_title;

const trialProg = trainingProgress(trialMLP);
trialProg.title = script.teaching.training_progress_title;

const trialSet: Dataset<number[][], string> = dataset("trialDataset", trialStore);


const $trialInstances = trialLabelSelect.$click
  .sample(cam.$images)
  .map(async (img) => ({
    type: 'image',
    data: img,
    x: await featureExtractor.process(img),
    y: trialLabelSelect.$items.value[trialLabelSelect.$selected.value],
    thumbnail: cam.$thumbnails.value,
  }))
  .awaitPromises<Instance<number[][], string>>()

$trialInstances.subscribe(trialSet.create.bind(trialSet));

const $trialResults = $features
  .map(features => trialMLP.predict(features))
  .awaitPromises<ClassifierResults>();

const $trialPredictions = $trialResults
  .filter(results => results !== null)
  .map(results => ({
      instanceId: '',
      label: results.label,
      confidences: results.confidences
  }))
  .awaitPromises<Prediction>();

const $trialSetChanges = trialSet.$changes.combine(
    (count: number, changes: any[]): [number, any[]] => [count, changes], 
    trialSet.$count
    );
$trialSetChanges.subscribe(async ([count, changes]) => {
  const instanceChanged = changes.some(change => change.level === 'instance');
  if (instanceChanged && count > 6) {
    time.startTimer({ countdown: data.teaching.duration });
    await trialMLP.train(trialSet);
  }
});

const trialLabelPrediction = classificationLabel($trialPredictions);
trialLabelPrediction.title = script.teaching.label_title;

const trialBrows = datasetBrowser(trialSet);
trialBrows.title = script.teaching.training_set_title;

const textUncertaintyChart = text({"text": ""});
textUncertaintyChart.title = script.teaching.uncertainty_title;

*/
// -----------------------------------------------------------
// USER TEST
// -----------------------------------------------------------
let permutedAccuracyImageIds = Utility.randomPermutation(accuracyTestImageUrls.length);
let permutedUncertaintyImageIds = Utility.randomPermutation(uncertaintyTestImageUrls.length);

const $accuracyTestImageIdx: Stream<number> = new Stream(0, true);
const $uncertaintyTestImageIdx: Stream<number> = new Stream(0, true);

const $permutedAccuracyImageIdx = $accuracyTestImageIdx.map(idx => permutedAccuracyImageIds[idx]);
const $permutedUncertaintyImageIdx = $uncertaintyTestImageIdx.map(idx => permutedUncertaintyImageIds[idx]);

const $accuracyTestImages: Stream<ImageData> = $permutedAccuracyImageIdx
  .map(async imageIdx => {
    const url = accuracyTestImageUrls[imageIdx];
    var image = await ImageUtility.fetchImage(url);
    {
      // crop the input to match the webcam.
      const w = data.webcam.input_width;
      const h = data.webcam.input_height;
      const dim = Math.min(w, h);
      image = image.crop({
        x: (image.width - dim) / 2,
        y: (image.height - dim) / 2,
        width: dim,
        height: dim,
      })
    }
    image.resize({
      width: data.image.display_width,
      height: data.image.display_height
    })
    return ImageUtility.getImageData(image);
  })
  .awaitPromises<ImageData>();
$accuracyTestImages.start();

$permutedAccuracyImageIdx.subscribe(imageIdx => {
  // const imageLabel = accuracyTestImageLabels[imageIdx];
  // accuracyTestText.$text.set(script.user_test.accuracy_title.replace('...', (script.misc.label_names as any)[imageLabel]));
});

const $uncertaintyTestImages: Stream<ImageData> = $permutedUncertaintyImageIdx
  .map(async imageIdx => {
    const url = uncertaintyTestImageUrls[imageIdx];
    var image = await ImageUtility.fetchImage(url);
    {
      // crop the input to match the webcam.
      const w = data.webcam.input_width;
      const h = data.webcam.input_height;
      const dim = Math.min(w, h);
      image = image.crop({
        x: (image.width - dim) / 2,
        y: (image.height - dim) / 2,
        width: dim,
        height: dim,
      })
    }
    image.resize({
      width: data.image.display_width,
      height: data.image.display_height
    })
    return ImageUtility.getImageData(image);
  })
  .awaitPromises<ImageData>();
$uncertaintyTestImages.start();

const accuracyTestDisplay = imageDisplay($accuracyTestImages);
accuracyTestDisplay.title = script.user_test.display_title;

const uncertaintyTestDisplay = imageDisplay($uncertaintyTestImages);
uncertaintyTestDisplay.title = script.user_test.display_title;

const accuracyTestBackButton = button({ text: script.user_test.back_button_text });
accuracyTestBackButton.title = script.user_test.back_button_title;

const uncertaintyTestBackButtons = Object.fromEntries(conditions
  .map(condition => {
    const but = button({ text: script.user_test.back_button_text });
    but.title = script.user_test.back_button_title;
    return [condition, but];
  })
);

// ACCURACY

const accuracyImageIdText = text({});
accuracyImageIdText.$text = $accuracyTestImageIdx.map((id) => `${id + 1} / ${accuracyTestImageUrls.length}`);
accuracyImageIdText.$text.value = `${$accuracyTestImageIdx.value + 1} / ${accuracyTestImageUrls.length}`;
accuracyImageIdText.title = script.user_test.index_title;

const uncertaintyImageIdText = text({});
uncertaintyImageIdText.$text = $uncertaintyTestImageIdx.map((id) => `${id + 1} / ${uncertaintyTestImageUrls.length}`);
uncertaintyImageIdText.$text.value = `${$accuracyTestImageIdx.value + 1} / ${accuracyTestImageUrls.length}`;
uncertaintyImageIdText.title = script.user_test.index_title;

const accuracyTestEntries = new Array<AccuracyTestEntry>(accuracyTestImageUrls.length);
const uncertaintyTestEntries = new Array<UncertaintyTestEntry>(uncertaintyTestImageUrls.length);

const accuracyTestText = text({});
accuracyTestText.title = '';
const accuracyTestSelect = buttonSelect(
  new Stream(Object.values(script.user_test.label_names), true),
  new Stream(Object.keys(script.user_test.label_names), true),
);
accuracyTestSelect.title = '';
accuracyTestSelect.title = script.user_test.accuracy_title;

// UNCERTAINTY

const uncertaintyTestSliders = Object.fromEntries(Object.entries(conditionUncertainties)
  .map(([condition, uncertainty]) => {
    const sliderArr = sliderArray(
      [script.misc.uncertainty_names[uncertainty] as string],
      [sliderOptions[condition as Condition]]
    );
    sliderArr.$values.set(new Array(Object.keys(script.misc.uncertainty_names).length).fill(0.5)),
      sliderArr.title = script.user_test.uncertainty_title;
    return [condition, sliderArr];
  })
);

const uncertaintyTestButtons = <Record<Condition, Button>>Object.fromEntries(conditions.map(condition => {
  const but = button({ text: script.user_test.answer_button_text });
  but.title = script.user_test.answer_title;
  return [condition, but];
}));

// -----------------------------------------------------------
// SURVEY
// -----------------------------------------------------------


const surveys = <Record<Condition, Survey>>Object.fromEntries(conditions
  .map(condition => {
    const surveyData = data.survey.data[condition as keyof typeof data.survey.data];
    const surv = survey({
      default: data.survey.data.common.default,
      ...surveyData
    } as SurveyData);
    surv.title = data.survey.survey_title;
    return [condition, surv];
  })
);
/*
const surveys: Record<Condition, Survey> = {
  'with_ambiguity': survey({
    default: data.survey.common.data.default,
    ...data.survey.with_ambiguity.data 
  } as SurveyData),
  'with_novelty': survey({
    default: data.survey.common.data.default,
    ...data.survey.with_novelty.data 
  }as SurveyData),
}
*/
const surveyButtons = <Record<Condition, Button>>Object.fromEntries(conditions
  .map(condition => {
    const but = button({ text: data.survey.survey_button_text });
    but.title = data.survey.survey_title;
    return [condition, but];
  })
);

// -----------------------------------------------------------
// DEBRIEFING
// -----------------------------------------------------------

const endResultAccuracyText = text({});
endResultAccuracyText.title = script.debriefing.result_accuracy_title;
const endResultUncertaintyText = text({});
endResultUncertaintyText.title = script.debriefing.result_uncertainty_title;

// -----------------------------------------------------------
// DEBUG
// -----------------------------------------------------------

const trainButton = button({ text: script.debug.train_button_text });
trainButton.title = script.debug.debug_title;

// -----------------------------------------------------------
// INTERACTIONS
// -----------------------------------------------------------


Object.entries(dashboards).forEach(([cond, dash]) => {
  dash.$previousPageName.subscribe(page => {
    const condition = cond as Condition;
    switch (page) {
      case script.misc.pages.teaching.name:
      case script.misc.pages.exploration.name:
        cam.stopCamera();
        break;
      case script.misc.pages.user_test_uncertainty.name:
        uncertaintyTestButtons[condition].$click.set(new CustomEvent(""));
        break;
      case script.misc.pages.user_test_accuracy.name:
        // force-click button in order to save!
        accuracyTestSelect.$click.set(new CustomEvent(""));
        break;
      case script.misc.pages.survey.name:
        // force-click survey button in order to save!
        surveyButtons[condition].$click.set(new CustomEvent(""));
        break;
      default:
        break;
    }
  });
  dash.$currentPageName.subscribe(async page => {
    // const participantId = $participantId.value;
    switch (page) {
      case script.misc.pages.teaching.name:
        time.stopTimer();
        break;
      case script.misc.pages.exploration.name:
        time.stopTimer();
        break;
      case script.misc.pages.user_test_accuracy.name:
        // quick fix: start the stream. For some reason start doesn't work.
        $accuracyTestImageIdx.set($accuracyTestImageIdx.value);
        break;
      case script.misc.pages.user_test_uncertainty.name:
        // quick fix: start the stream. For some reason start doesn't work.
        $uncertaintyTestImageIdx.set($uncertaintyTestImageIdx.value);
        break;
      case script.misc.pages.survey.name:
        break;
      case script.misc.pages.debriefing.name:
        // endWiz.show();
        let accuracyData;
        {
          const conditionRequests = await Promise.all(Object.values(accuracyTestServices).map(serv => force(() => serv.find({}))));
          // take last message for each condition.
          const messages = conditionRequests.map(res => res.data[res.data.length - 1] as unknown as Message<AccuracyTestEntry[]>);
          // sort by iteration.
          messages.sort((a, b) => a.content.task.iteration >= b.content.task.iteration ? 1 : -1);

          accuracyData = messages.map(msg => ({
            task: msg.content.task,
            score: getAccuracyScore(msg.content.task.condition as Condition, msg.content.task.iteration, msg.content.data)
          }));
        }
        const accuracyTestResultText = accuracyData.reduce((acc, data, idx) =>
          acc + `<div><p>N°${data.task.iteration + 1}, ${data.task.condition}</p>
            <ul>
            <li>Note : ${(data.score.mark).toPrecision(2)} / 20</li>
            <li>${data.score.numberUnderestimate} sous-estimations</li>
            <li>${data.score.numberOverestimate} surestimations</li>
            </ul>
            </div>`,
          ''
        );
        endResultAccuracyText.$text.set(accuracyTestResultText);
        break;
      default:
        break;
    }
  });
});


timerButton.$click.subscribe(() => {
  const condition = $condition.value;
  const currentPage = dashboards[condition].$currentPageName.value;
  const timeState = time.$time.value.state;
  switch (timeState) {
    case 'running':
      time.pauseTimer(true);
      break;
    case 'paused':
      time.pauseTimer(false);
      break;
    default:
    case 'stopped':
      time.startTimer({
        countdown: currentPage === script.misc.pages.teaching.name
          ? data.teaching.duration
          : data.exploration.duration
      });
      break;
  }
});


const $budget = new Stream(0, true);

$condition.subscribe(condition => {
  deepEnsemble.$training.subscribe(async (status) => {
    const notTrainingStatus = ['success', 'error', 'idle'];
    const isTraining = !notTrainingStatus.includes(status.status);
    const justFinishedTraining = status.status === 'success';
    //
    // todo: use Stream.combineArray instead (or zipArray, actually).
    // see: https://mostcore.readthedocs.io/en/latest/api.html#merge
    const id = $participantId.value;
    const budget = $budget.value;
    //
    const numberClasses = trainingLabelSelect.$disabled.value.length;
    const disabled = new Array(numberClasses).fill(isTraining);
    trainingLabelSelect.$disabled.set(disabled);
    //
    // time.pauseTimer(isTraining);
    //
    if (isTraining) {
      labelPrediction.displayLabel.$text.set(script.teaching.prediction_pending_text);
      labelPrediction.displayLabel.$text.stop();
      //
      trainingUncertaintyCharts[condition].$data.set([]);
      trainingUncertaintyCharts[condition].$data.stop();
    }
    else {
      labelPrediction.displayLabel.$text.start();
      trainingUncertaintyCharts[condition].$data.start();
    }
    if (justFinishedTraining) {
      // save
      const deepEnsembleDataId = `${getIdService(id, condition)}_${budget}_deepensemble`;
      await Promise.all([deepEnsemble.save(deepEnsembleDataId), /* gmm.save(gmmDataId) */]);
    }
  });
});

// ANSWER SUBMISSION
const $accuracyTestData = Stream.combineArray(
  (condition: Condition, participantId: string, iteration: number, imageIdx: number, permutedImageIdx: number, testImage: ImageData, selected: string)
    : [Condition, string, number, number, number, ImageData, string] =>
    [condition, participantId, iteration, imageIdx, permutedImageIdx, testImage, selected],
  [$condition, $participantId, $iteration, $accuracyTestImageIdx, $permutedAccuracyImageIdx, $accuracyTestImages, accuracyTestSelect.$selected as Stream<string>]
);
accuracyTestSelect.$click
  .sample($accuracyTestData)
  .subscribe(async ([condition, participantId, iteration, imageIdx, permutedImageIdx, image, selected]) => {
    if (selected == null) {
      notification({
        title: script.misc.notifications.error.title,
        message: script.user_test.error.answer.text,
        duration: 5000,
        type: 'danger'
      });
      return;
    }
    const x = await featureExtractor.process(image);
    const pred = await deepEnsemble.predict(x);

    const predicted = pred.label;
    const truth = accuracyTestImageLabels[permutedImageIdx];

    const isClassifierRight = predicted === truth;
    let result: AccuracyTestResult;
    switch (selected) {
      case 'dont_know':
        result = 'dont_know';
        break;
      case 'yes':
        result = isClassifierRight ? 'right' : 'over';
        break;
      case 'no':
        result = isClassifierRight ? 'under' : 'right';
        break;
      default:
        logger.error(`Answer not known: ${selected}`);
        break;
    }

    const accuracyEntry: AccuracyTestEntry = {
      answer: {
        value: selected
      },
      labels: {
        predicted: predicted,
        truth: truth
      },
      result: result,
      image: {
        idx: imageIdx,
        order: permutedImageIdx,
        name: accuracyTestImageUrls[imageIdx]
      }
    };
    accuracyTestEntries[permutedImageIdx] = accuracyEntry;

    // cleaning up
    // accuracyTestSelect.$selected.set(null);

    // Switching to next image from the set.
    // todo: log answers.
    const idx = imageIdx + 1;
    if (idx < accuracyTestImageUrls.length) {
      $accuracyTestImageIdx.set(idx);
    }
    else {
      // save
      await force(() =>
        MyService.save(
          accuracyTestServices[condition],
          participantId,
          condition,
          iteration,
          data.tasks.user_test_accuracy,
          accuracyTestEntries)
      );
      notification({
        title: script.misc.notifications.save.title,
        message: script.misc.notifications.save.description,
        duration: 5000,
        type: 'default'
      });
      // dashboards[condition].$currentPageName.set(script.misc.pages.survey.name);
    }
  });


const $uncertaintyTestConfirm = Object.fromEntries(conditions
  .map(condition => {
    const $confirm = Stream.combineArray(
      (participantId: string, iteration: number, imageIdx: number, permutedImageIdx: number, testImage: ImageData, sliderValues: number[])
        : [string, number, number, number, ImageData, number[]] =>
        [participantId, iteration, imageIdx, permutedImageIdx, testImage, sliderValues],
      [$participantId, $iteration, $uncertaintyTestImageIdx, $permutedUncertaintyImageIdx, $uncertaintyTestImages, uncertaintyTestSliders[condition].$values]
    );
    return [condition, $confirm];
  })
);

Object.entries(uncertaintyTestButtons).forEach(([cond, but]) => {
  const condition = cond as Condition;
  const uncertainty = conditionUncertainties[condition];
  but.$click.sample($uncertaintyTestConfirm[condition])
    .subscribe(async ([participantId, iteration, imageIdx, permutedImageIdx, image, sliderValues]) => {
      // cleaning up 
      // fixme: slider values are not updated in the SliderArray svelete component.
      // This is internal to range-slider-pips, apparently.
      // uncertaintyTestSliders[condition].$values.set(new Array(sliderValues.length).fill(0.2));

      // Adding the score
      {
        const x = await featureExtractor.process(image);
        const [ensembleRes, densityRes] = await Promise.all([deepEnsemble.predict(x), densityClassifier.predict(x)]);
        const truth = UncertaintyUtility.computeUncertainty(ensembleRes, densityRes);
        const answer = {
          [uncertainty]: sliderValues[0]
        } as Partial<Uncertainty>;
        uncertaintyTestEntries[permutedImageIdx] = {
          answer: answer,
          truth: truth,
          uncertainty: uncertainty,
          image: {
            idx: imageIdx,
            order: permutedImageIdx,
            name: uncertaintyTestImageUrls[imageIdx]
          }
        };
      }

      // Switching to next image from the set.
      const idx = imageIdx + 1;
      if (idx < uncertaintyTestImageUrls.length) {
        $uncertaintyTestImageIdx.set(idx);
      }
      else {
        // save
        await force(() =>
          MyService.save(
            uncertaintyTestService,
            participantId,
            condition,
            iteration,
            data.tasks.user_test_uncertainty,
            uncertaintyTestEntries
          )
        );
        notification({
          title: script.misc.notifications.save.title,
          message: script.misc.notifications.save.description,
          duration: 5000,
          type: 'default'
        });
        // dashboards[condition].$currentPageName.set(script.misc.pages.user_test_accuracy.name);
      }
    });
});

const $accuracyBackButtonIdx = accuracyTestBackButton.$click.sample($accuracyTestImageIdx);
$accuracyBackButtonIdx.subscribe((imageIdx) => {
  const previousIdx = Math.max(0, imageIdx - 1);
  if (accuracyTestEntries[previousIdx] !== undefined) {
    // todo: telegraph the selected button (show that it is selected!)
    accuracyTestSelect.$selected.set(accuracyTestEntries[previousIdx].answer.value);
  }
  $accuracyTestImageIdx.set(previousIdx);
});

const $uncertaintyBackButtonIds = Object.fromEntries(conditions
  .map(condition => {
    const $click = uncertaintyTestBackButtons[condition].$click.sample($uncertaintyTestImageIdx);
    return [condition, $click];
  })
);
Object.entries($uncertaintyBackButtonIds).forEach(([condition, $imageIdx]) => {
  $imageIdx.subscribe(imageIdx => {
    const previousIdx = Math.max(0, imageIdx - 1);
    if (uncertaintyTestEntries[previousIdx] !== undefined) {
      uncertaintyTestSliders[condition].$values.set(Object.values(uncertaintyTestEntries[previousIdx].answer));
    }
    $uncertaintyTestImageIdx.set(previousIdx);
  });
})

// SURVEY 

const $surveyAnswers = <Record<Condition, Stream<[string, number, SurveyEntry[]]>>>Object.fromEntries(conditions
  .map(condition => {
    const $participantEntries = Stream.combineArray(
      (participantId: string, iteration: number, entries: SurveyEntry[]): [string, number, SurveyEntry[]] => [participantId, iteration, entries],
      [$participantId, $iteration, surveys[condition].$entries]
    );
    const $click = surveyButtons[condition].$click.sample($participantEntries);
    return [condition, $click];
  })
);
Object.entries($surveyAnswers).forEach(([cond, $surveyAnswer]) => {
  const condition = cond as Condition;
  $surveyAnswer.subscribe(async ([participantId, iteration, entries]) => {
    // TODO
    await force(() =>
      MyService.save(
        surveyService,
        participantId,
        condition,
        iteration,
        data.tasks.survey,
        entries
      )
    );
    notification({
      title: script.misc.notifications.save.title,
      message: script.misc.notifications.save.description,
      duration: 5000,
      type: 'default'
    });
    // dashboards[condition].$currentPageName.set(script.misc.pages.debriefing.name);
  });
})


// -----------------------------------------------------------
// LAUNCH
// -----------------------------------------------------------

const $homeData = Stream.combineArray(
  (condition: Condition, participantId: string, iteration: number): [Condition, string, number] => [condition, participantId, iteration],
  [$condition, $participantId, $iteration]
);

homeButton.$click
  .sample($homeData)
  .subscribe(async ([condition, participantId, iteration]) => {
    if (!participantId.match(new RegExp(data.participants.expr))) {
      notification({
        title: script.misc.notifications.error.title,
        message: script.home.error.participant_id.text,
        duration: 5000,
        type: 'danger'
      });
      return;
    }
    await Promise.all([gmmConnectionPromise, storeConnectionPromise])
    await densityClassifier.reset();
    // Marcelle will prepend 'dataset' to the service's id automatically.
    const setAndConnectDataset = async () => {
      // Force to wait for the connection to server (data set).
      set = dataset(getIdService(participantId, condition), store);
      await set.ready;
      brows = datasetBrowser(set);
      brows.title = script.teaching.training_set_title;
    }
    await force(setAndConnectDataset);
    //
    const $inputInstances = trainingLabelSelect.$click
      .sample(cam.$images)
      .map(async (img) => ({
        type: 'image',
        data: img,
        thumbnail: cam.$thumbnails.value,
        x: await featureExtractor.process(img),
        y: trainingLabelSelect.$selected.value,
        timestamp: new Date()
      })
      )
      .awaitPromises<Instance<number[][], string>>();
    $inputInstances.subscribe(set.create.bind(set));
    //
    const $trainingSetChanges = set.$changes.combine(
      (count: number, changes: any[]): [number, any[]] => [count, changes],
      set.$count
    );

    const trainModels = async () => Promise.all([deepEnsemble.train(set), densityClassifier.train(set)]);
    $trainingSetChanges.subscribe(async ([count, changes]) => {
      const instanceChanged = changes.some(change => change.level === 'instance');
      const timeState = time.$time.value.state;
      if (instanceChanged && count >= 3) {
        if (timeState == 'stopped') {
          time.startTimer({ countdown: data.teaching.duration });
        }
        await trainModels();
      }
    });
    trainButton.$click.subscribe(trainModels);
    //
    accuracyTestServices = <Record<Condition, MyService<AccuracyTestEntry[]>>>Object.fromEntries(conditions.map(c =>
      [c, myService<AccuracyTestEntry[]>(store, `${data.services.user_test_accuracy}-${getIdService(participantId, c)}`)
      ]
    ));
    uncertaintyTestService = myService(store, `${data.services.user_test_uncertainty}-${getIdService(participantId, condition)}`);
    surveyService = myService(store, `${data.services.survey}-${getIdService(participantId, condition)}`);
    //
    Object.entries(dashboards).forEach(([cond, dash]) => {
      const condition = cond as Condition;
      dash.settings
        .dataStores(store)
        .datasets(set)
        .models(deepEnsemble);
      if (iteration === 0) {
        dash
          .page(script.misc.pages.introduction.name, false).use(videoPrimer);

        dash
          .page(script.misc.pages.instructions.name, false)
          .use(videoInterface);
      }
      dash
        .page(script.misc.pages.teaching.name)
        .sidebar(cam, trainingLabelSelect, time, timerButton, trainingProg)
        .use(brows, [labelPrediction, trainingUncertaintyCharts[condition]]);

      dash
        .page(script.misc.pages.exploration.name)
        .sidebar(cam, time, timerButton)
        .use(brows, [labelPrediction, trainingUncertaintyCharts[condition]]);

      dash
        .page(script.misc.pages.user_test_uncertainty.name)
        .sidebar(uncertaintyTestDisplay, uncertaintyImageIdText)
        .use(uncertaintyTestSliders[condition], uncertaintyTestButtons[condition], uncertaintyTestBackButtons[condition]);

      dash
        .page(script.misc.pages.user_test_accuracy.name)
        .sidebar(accuracyTestDisplay, accuracyImageIdText)
        .use(accuracyTestSelect, accuracyTestBackButton);
      dash
        .page(script.misc.pages.survey.name,
          false // no sidebar.
        ).use(surveys[condition], surveyButtons[condition]);

      if (iteration === 0) {
        dash
          .page(script.misc.pages.break.name,
            false // no sidebar.
          );
      }
      else if (iteration === 1) {
        dash
          .page(script.misc.pages.debriefing.name,
            false // no sidebar.
          ).use(endResultAccuracyText);
      }

      dash.page(script.misc.pages.debug.name)
        .sidebar(trainButton);

    });
    //

    /*
    dashboards['with']
    .page(script.misc.pages.trial.name)
    .sidebar(cam, trialLabelSelect, trialProg)
    .use(trialBrows, [trialLabelPrediction, textUncertaintyChart]);
    */

    // dashboards[condition].$currentPageName.set(script.misc.pages.introduction.name);
    dashboards[condition].show();
    homeDashboard.hide();
  });

homeDashboard.show();
