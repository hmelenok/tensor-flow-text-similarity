import  '@tensorflow/tfjs-node';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import {readFileSync} from 'fs';

const ticketsFile = readFileSync('./dataset/tickets', 'utf-8');
const input_sentences = ticketsFile.split('\n').filter(Boolean);
const input_threshold = 0.65;

const plotly_heatmap = {data:[], layout:{}};
const output_resultshtml = "";
const analyzing_text = true;


async function onClickAnalyzeSentences(){

    const list_sentences = [];
    for(const i in input_sentences){
        if(input_sentences[i].length){
            list_sentences.push(input_sentences[i]);
        }
    }

    // console.log({list_sentences});
    get_similarity(list_sentences);
}

async function get_embeddings(list_sentences) {
    const model = await use.load();
    return await model.embed(list_sentences)
}

function dot(a, b){
    const hasOwnProperty = Object.prototype.hasOwnProperty;
    let sum = 0;
    for (const key in a) {
        if (hasOwnProperty.call(a, key) && hasOwnProperty.call(b, key)) {
            sum += a[key] * b[key]
        }
    }
    return sum
}

function similarity(a, b) {
    const magnitudeA = Math.sqrt(dot(a, a));
    const magnitudeB = Math.sqrt(dot(b, b));
    if (magnitudeA && magnitudeB)
        return dot(a, b) / (magnitudeA * magnitudeB);
    else return false
}

function cosine_similarity_matrix_fn(matrix){
    let cosine_similarity_matrix = [];
    for(let i=0;i<matrix.length;i++){
        let row = [];
        for(let j=0;j<i;j++){
            row.push(cosine_similarity_matrix[j][i]);
        }
        row.push(1);
        for(let j=(i+1);j<matrix.length;j++){
            row.push(similarity(matrix[i],matrix[j]));
        }
        cosine_similarity_matrix.push(row);
    }
    return cosine_similarity_matrix;
}

function form_groups(cosine_similarity_matrix){
    let dict_keys_in_group = {};
    let groups = [];

    for(let i=0; i<cosine_similarity_matrix.length; i++){
        const this_row = cosine_similarity_matrix[i];
        for(let j=i; j<this_row.length; j++){
            if(i!=j){
                let sim_score = cosine_similarity_matrix[i][j];

                if(sim_score > input_threshold){

                    let group_num;

                    if(!(i in dict_keys_in_group)){
                        group_num = groups.length;
                        dict_keys_in_group[i] = group_num;
                    }else{
                        group_num = dict_keys_in_group[i];
                    }
                    if(!(j in dict_keys_in_group)){
                        dict_keys_in_group[j] = group_num;
                    }

                    if(groups.length <= group_num){
                        groups.push([]);
                    }
                    groups[group_num].push(i);
                    groups[group_num].push(j);
                }
            }
        }
    }

    let return_groups = [];
    for(const i in groups){
        return_groups.push(Array.from(new Set(groups[i])));
    }

    // console.log({return_groups});
    return return_groups;
}

async function get_similarity(list_sentences){

    let makeGroups = function(embeddings) {


        let cosine_similarity_matrix = cosine_similarity_matrix_fn(embeddings.arraySync());


        let groups = form_groups(cosine_similarity_matrix);

        let html_groups = "";
        for(let i in groups){
            html_groups+="<br/><b>Group "+String(parseInt(i)+1)+"</b><br/>";
            for(let j in groups[i]){
                html_groups+= list_sentences[ groups[i][j] ] + "<br/>";
            }
        }

        return groups.map(group => {
            return group.map(sentence_index => list_sentences[sentence_index]);
        });
    };

    let embeddings = await get_embeddings(list_sentences);
    console.warn({embeddings2: embeddings, groups: makeGroups(embeddings)})
}

onClickAnalyzeSentences();