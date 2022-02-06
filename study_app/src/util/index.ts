
export * from './uncertainty';
export * from './multivariate-normal';
export * from './image';

export interface ImageSize {
  width: number,
  height: number
}

export class Utility {

  public static argMaxObj(obj: Object): any {
    // inspired by: https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28#gistcomment-3202905
    return Object.entries(obj)
      .reduce((acc, val) => (val[1] > acc[1] ? val : acc))[0];
  }

  public static argMaxArr(array: Array<number>): number {
    // taken from: https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28#gistcomment-3202905
    return [].reduce.call(array, (m, c: number, i: number, arr: number[]) => c > arr[<number>m] ? i : m, 0) as number;
  }

  /**
   * 
   * @param min 
   * @param max 
   * @returns random integer in [min, max] (inclusive)
   */
  public static randomInt(min: number, max: number): number {
    // taken from: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random#getting_a_random_integer_between_two_values_inclusive
    min = Math.ceil(min)
    max = Math.floor(max)
    return Math.floor(min + Math.random() * (max - min + 1))
  }

  /**
   * 
   * @param n 
   * @returns random permutation of [0, n-1] (inclusive)
   */
  public static randomPermutation(n: number): Array<number> {
    // todo
    const remaining = new Array(n).fill(0).map((_, idx) => idx);
    const arr = new Array(n);
    for (let i = 0; i < n; ++i) {
      const last = n-i-1;
      const j = this.randomInt(0, last);
      //
      const value = remaining[j];
      // swap value for current last, (no longer use value)
      remaining[j] = remaining[last];
      //
      arr[i] = value;
    }
    return arr;
  }

  public static shuffle<T>(arr: Array<T>): Array<T> {
    const permutation = this.randomPermutation(arr.length);
    const copy = [...arr];
    copy.map((_, idx) => arr[permutation[idx]]); 
    return copy;
  }

}
